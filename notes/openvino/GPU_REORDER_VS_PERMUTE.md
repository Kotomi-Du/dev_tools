# Intel GPU: Reorder vs. Permute

## Main distinction

Both operations can move data in memory, but their tensor semantics differ:

- **Reorder** changes how a tensor is stored, including physical memory layout
  and/or element type. A layout-only reorder preserves logical coordinates and
  logical axis sizes.
- **Permute** changes the ordering of logical axes. The output shape and
  coordinates follow the specified permutation.

| Property | Reorder | Permute |
|---|---|---|
| Primary purpose | Storage-layout or data-type conversion | Logical axis transposition |
| Typical OpenVINO operators | `Convert`, `ConvertLike` | `Transpose` |
| GPU primitive | `cldnn::reorder` | `cldnn::permute` |
| General OpenCL kernel | `reorder_data.cl` | `permute_ref.cl` |
| Logical shape | Normally unchanged | Axis sizes reordered |
| Physical memory | Can change | Can change |

These are their primary roles, not absolute capability limits: graph
optimizations can fuse operations, and a permute implementation can incorporate
output-layout or data-type conversion.

## Why `bfyx` to `yxfb` is a reorder

Here `b`, `f`, `y`, and `x` identify batch, feature, height, and width.
The format specifies physical storage order, from outermost to innermost:

- `bfyx`: `x` is contiguous.
- `yxfb`: `b` is contiguous.

A layout-only reorder preserves the value at each logical coordinate:

```text
Input logical coordinate:  (b=1, f=2, y=3, x=4)
Output logical coordinate: (b=1, f=2, y=3, x=4)
```

Its memory offset changes, but the meanings and sizes of the logical axes do
not. For contiguous, unpadded storage with sizes `B`, `F`, `Y`, and `X`:

```text
bfyx offset = ((b * F + f) * Y + y) * X + x
yxfb offset = ((y * X + x) * F + f) * B + b
```

Thus, changing storage order is not the same as transposing logical axes.

## What permute changes

For a logical input shape `[2, 3, 4, 5]` in `[b, f, y, x]` order:

```text
Permutation:  {1, 0, 2, 3}
Output shape: [3, 2, 4, 5]

Input coordinate:  (b=1, f=2, y=3, x=4)
Output coordinate: (b=2, f=1, y=3, x=4)
```

Batch and feature axes exchange roles. Both input and output may still use the
`bfyx` storage format.

In contrast, converting FP16 to UINT2 without transposing keeps the logical
shape `[2, 3, 4, 5]` and changes the element representation.

## Operator mapping and kernel selection

### Transpose

On the Intel GPU OpenCL path:

```text
ov::op::v1::Transpose
    -> cldnn::permute
    -> permute kernel selector
    -> an eligible optimized kernel, or permute_ref
```

Candidates include:

- `permute_tile_8x8_4x4`
- `permute_tile_8x8_4x4_fsv`
- `permute_bfzyx_to_bfyxz`
- `permute_bf_swap`
- `permute_f_y_axes`
- `permute_xy_swap`
- `permute_ref` as the general fallback

The selected kernel depends on supported types, layouts, shape, permutation,
offsets, pitches, fused operations, and kernel priority.

In [PR #38206](https://github.com/openvinotoolkit/openvino/pull/38206),
`permute_bf_swap` has high priority for UINT2 batch/feature swapping with order
`{1, 0, 2, 3}` when:

- Input and output are UINT2 in `bfyx`.
- Shapes are static.
- There are no fused operations.
- Pitches match logical dimensions and first-element offsets are zero.
- The spatial plane `H * W` is divisible by 64 elements, or 16 packed bytes.

This PR does not add UINT2 support to `permute_ref`. An unaligned UINT2 plane
has no eligible permute kernel and network construction fails with
"Could not find a suitable kernel"; it does not fall back to `permute_ref`.
Other data types, such as FP32, can still select another supported kernel.

### Convert and ConvertLike

```text
Convert / ConvertLike
    -> cldnn::reorder
    -> reorder kernel selector
    -> an eligible specialized kernel, or reorder_data
```

- `Convert` requests a destination element type.
- `ConvertLike` requests the element type of its second input.
- The GPU compiler can also insert reorders between layers to satisfy physical
  format requirements, or to adapt input/output storage.

A reorder primitive does not necessarily execute `reorder_data.cl`: specialized
reorder kernels also exist. Weight reordering has a separate selector.

Graph optimization can remove or fuse either kind of operation, so a model
operator does not guarantee a standalone kernel launch.

## PR #38206: grouped UINT2 weights for FullyConnectedCompressed

[PR #38206](https://github.com/openvinotoolkit/openvino/pull/38206) adds support
for compressed weights stored as `[num_group, N, group_size]`. The existing
FullyConnectedCompressed layouts are `[N, num_group * group_size]` and
`[num_group * group_size, N]`.

The targeted conversion is:

```text
[num_group, N, group_size]
    -> Transpose: swap the first two axes
[N, num_group, group_size]
    -> Reshape: merge the group dimensions
[N, num_group * group_size]
    -> FullyConnectedCompressed
```

The related changes are:

- **Pattern recognition:** recognize decompression paths with
  `Transpose -> Reshape -> FC`, in addition to existing patterns. Preserve
  scale/zero-point layouts when they already match the compressed FC's
  requirements, applying only the necessary reshape/squeeze.
- **Packed UINT2 permute:** add and register `permute_bf_swap` for the aligned
  `bfyx` case described above. Each work-item copies 16 packed bytes from
  plane `(b, f)` to `(f, b)` without unpacking/repacking individual values.
- **Deliberately limited scope:** no general UINT2 reference-permute support
  and no handling of the separate
  `[G, N, S] -> [G, S, N] -> [G, S * N]` pattern.
- **Tests:** cover transformation matching, aligned fast-path selection,
  unaligned UINT2 rejection, FP32 fallback, and packed-output correctness.
  The correctness test uses distinct batch/feature plane patterns and checks
  against independently computed expected bytes, not a UINT2 reference kernel.

Although the PR describes "reordering weights," the axis exchange is a
**permute**, not a layout-only `cldnn::reorder`: the logical shape changes,
while the GPU input and output can both remain `bfyx`.

## UINT2 packing: general reorder vs. specialized permute

The general `reorder_data.cl` kernel supports:

- **Input decoding:** four 2-bit elements per byte.
- **Output packing:** sixteen 2-bit elements per 32-bit word.
- **Atomic packed writes:** atomic AND clears the destination field and atomic
  OR sets its new value without overwriting neighboring fields.

For logical element index `i`:

```text
Input byte index = i / 4
Input bit shift  = (i % 4) * 2
Decoded value    = (input[byte index] >> bit shift) & 3

Output word index = i / 16
Output bit shift  = (i % 16) * 2
```

Reorder computes `i` by mapping logical coordinates between source and
destination layouts. In contrast, `permute_bf_swap` moves whole packed spatial
blocks between batch/feature planes; it needs neither per-element decoding
nor atomic packed writes. `permute_ref.cl` does not support UINT2 in this PR.


- [Transpose to permute mapping](../pr2-openvino/src/plugins/intel_gpu/src/plugin/ops/transpose.cpp)
- [Convert and ConvertLike to reorder mapping](../pr2-openvino/src/plugins/intel_gpu/src/plugin/ops/convert.cpp)
- [Permute OpenCL implementation and order conversion](../pr2-openvino/src/plugins/intel_gpu/src/graph/impls/ocl/permute.cpp)
- [Permute kernel selector](../pr2-openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_selector.cpp)
- [UINT2 batch/feature swap validation](../pr2-openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/permute/permute_kernel_bf_swap.cpp)
- [Reorder kernel selector](../pr2-openvino/src/plugins/intel_gpu/src/kernel_selector/kernels/reorder/reorder_kernel_selector.cpp)
- [General reorder kernel](../pr2-openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/reorder_data.cl)
- [Reference permute kernel](../pr2-openvino/src/plugins/intel_gpu/src/kernel_selector/cl_kernels/permute_ref.cl)
- [UINT2 storage allocation size](../pr2-openvino/src/plugins/intel_gpu/include/intel_gpu/runtime/layout.hpp)
- [Reorder unit tests](../pr2-openvino/src/plugins/intel_gpu/tests/unit/test_cases/reorder_gpu_test.cpp)
- [Permute unit tests](../pr2-openvino/src/plugins/intel_gpu/tests/unit/test_cases/permute_gpu_test.cpp)
