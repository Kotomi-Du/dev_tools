# GPU FullyConnectedCompressed: skip scale/ZP transpose (`transpose_b_scale` / `transpose_b_zp`)

Branch: `refactor_fccompressed`

## 1. Design of [PR #38697](https://github.com/openvinotoolkit/openvino/pull/38697])

**Commit title:** "skip transpose of scale/zp if scale/zp layout is as kernel expected"

### Motivation
`ConvertFullyConnectedToFullyConnectedCompressed` used to handle weights stored as `[K, N]` with a `Transpose` in front of the FC by inserting a `Transpose` node on the scale and the ZP as well. These extra nodes converted `[groups, N]` into `[N, groups]`. Later, `prepare_quantization` reordered the scale/ZP again so that N became the innermost dimension, which is what the kernels expect. Data that already had N innermost was therefore transposed and then transposed back, at the cost of extra nodes and memory traffic.

### Core idea
Keep scale/ZP in their original layout. Each FC now carries two per-parameter flags that describe that layout:

| Flag | `true` (default, old behavior) | `false` (new) |
|---|---|---|
| `transpose_b_scale` | scale is `[N, groups]` | scale is `[groups, N]` (N innermost) |
| `transpose_b_zp` | ZP is `[N, groups]` | ZP is `[groups, N]` |

### Changes by layer

**1. `op::FullyConnectedCompressed`** (`op/fully_connected_compressed.hpp/.cpp`)
- New members `m_transpose_b_scale` and `m_transpose_b_zp`, both defaulting to `true`.
- All three constructors gained the two trailing parameters.
- `clone_with_new_inputs` propagates both flags.
- New `visit_attributes` override that serializes both flags, so they show up in IR dumps and attribute comparison.

**2. `ConvertFullyConnectedToFullyConnectedCompressed`**
- The `Transpose` that used to be cloned onto scale/ZP has been removed.
- The flags are set to `transpose_b_scale = transpose_b_zp = !has_transpose`.
- Any ZP `Convert` is now added to `result_nodes` so that its runtime info is copied.

**3. Other transformations that rebuild the FC**
- `FullyConnectedConvertFusion`, `DynamicQuantizeFullyConnected` and `FullyConnectedHorizontalFusion` now forward both flags to the new FC.

**4. `cldnn::fully_connected` primitive** (`primitives/fully_connected.hpp`)
- New fields `transpose_b_scale` and `transpose_b_zp`.
- They are included in hashing, `==`, `save` and `load`.
- The plugin op builder (`plugin/ops/fully_connected.cpp`) passes them through.

**5. `prepare_quantization::optimize_weights_decompression_parameters`**
- `need_reorder(...)` now returns `false` when the parameter is not transposed, so no reorder is inserted because the layout already has N innermost.

**6. OCL impl** (`impls/ocl/fully_connected.cpp`)
- New `convert_decompression_tensor()`: when the tensor is `bfyx`, not transposed, and its feature dimension equals the weights' OFM, it is reinterpreted as `fbyx`. Existing kernels then still address batch as N and feature as groups, so no kernel changes are needed.
- Typo fix: `commpressed` → `compressed`.

**7. oneDNN impl** (`impls/onednn/fully_connected_onednn.cpp`)
- The group count (`ngroups`) for scale/ZP is now read from an axis chosen by the flag through `get_ifm_dim_idx(layout, transpose_b_param, ...)`, instead of a hardcoded axis.
- `save`/`load` serialize the flags so model caching works.

**8. Tests**
- New unit test `fully_connected_gpu_tests.compressed_int8_transpose_scale_zp`: u8 weights `[3, 32]`, scale/ZP `[2, 3]` (`[groups, N]`), both flags `false`, with outputs checked against expected values.

### Data flow (non-transposed case)
```
Weights [K,N] --Transpose--> FC (transpose_b=true)
Scale   [G,N] -------------> FC (transpose_b_scale=false)   // no Transpose, no reorder
ZP      [G,N] -------------> FC (transpose_b_zp=false)
   OCL : bfyx[G,N] reinterpreted as fbyx (N=batch, G=feature)
   oneDNN: ngroups read from axis 0 ([G,N])
```

## 2. Follow-up fixes from review

### 2.1 oneDNN `get_ifm_dim_idx` (`fully_connected_onednn.cpp`)
- **Problem:** the axis was chosen by counting non-unit dimensions. For a non-transposed per-channel parameter `[1, N]` that count is 1, so the code read `N` as the group count. This could produce a zero group size or wrong grouped attributes.
- **Fix:**
  - The function now takes `weights_rank` and derives the axis from `transpose_b_param`: last axis for `[N, G]`, second-to-last for `[G, N]`.
  - Unit dimensions keep their positions.
  - Extra dimensions count as legacy trailing padding if they are all 1s, and as leading broadcast dimensions otherwise.
  - A 1D `[N]` shape returns index 1, which reads the padded 1, meaning one group.
- **Call sites:** `create()` and `load()`, for both scale and ZP, pass the weights rank.

### 2.2 Horizontal FC fusion (`fc_horizontal_fusion.cpp`)
- **Problem:** scale/ZP were always concatenated on axis 0. For the `[G, N]` layout, Concat validation fails when Q/K/V have different `N`, and the result is wrong when they are equal.
- **Fix:**
  - New `get_param_n_axis()` returns axis 0 when the parameter is transposed and the last axis otherwise. It is used for the fused scale and for non-scalar ZPs.
  - Fusion is rejected unless all FCs share the same `transpose_b`, `transpose_b_scale` and `transpose_b_zp`.
- **After merging master:** master now folds concats via `concat_and_fold`, which takes an `OutputVector`.
  - `get_param_n_axis` now takes an `ov::Output<ov::Node>`.
  - An extra `copy_runtime_info` call that had the wrong type and ran before the null check was removed. `concat_and_fold` already copies runtime info, which fixes the `fused_names_0` failure.

### 2.3 `convert_fc_to_compressed_test.cpp`
- In tests 6–9 and 13–15, the expected models now pass `true, false, false` instead of `true, 0, 0`.
- `comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES)` is enabled only in those 7 tests, so the flags are actually verified. `TransformationTestsF` does not compare attributes by default.

### 2.4 `horizontal_fc_fusion_test.cpp`
- **`FullyConnectedHorizontalFusion_non_transposed_scale_zp`:** three FCs with N = 1024, 512 and 128 and `[32, N]` scale/ZP. The expected result is folded Constants: weights `[1664, 4096]`, scale and ZP `[32, 1664]`.
- **`FullyConnectedHorizontalFusion_mixed_scale_orientation_no_fusion`:** mixed scale orientation must not fuse.

### 2.5 Notes on `is_caching_test`
`get_network(..., is_caching_test=true)` serializes the compiled program to a blob and loads it into a new network. This exercises `save()`/`load()`, including the oneDNN path that recomputes scale/ZP attributes from the serialized flags.

### 2.6 Regression: per-expert zero points treated as ordinary per-channel zero points

**Affected topology:**
- Weights: `[64, N, K]`, with `transpose_b = true`.
- Zero points: `[64, N, 1]`, with `transpose_b_zp = true`.
- Assume `N > 1` and `K > 1`, so the weights' non-unit dimension count is 3.
- Axis 0 selects the expert, axis 1 selects the output channel, and the final singleton axis means one zero point across all K values for that expert/channel pair.

#### Before the PR: correct expert handling by accident

The old code chose the zero-point IFM axis using the number of non-unit zero-point dimensions:

```cpp
// Zero-point shape: [64, N, 1]
dzp_rank = count_if(dims > 1);             // 2, because the trailing 1 is ignored
dzp_ifm_dim_idx = weights_transposed
                     ? dzp_rank - 1
                     : dzp_rank - 2;     // 1
ngroups = dzp_layout.get_dim(1);          // N, not the actual group count of 1
```

Because `ngroups != 1`, the code selected the grouped branch:

```cpp
attr->set_zero_points(DNNL_ARG_WEIGHTS, grouped, {group_size, 1}, dzp_data_type);
```

For a 3D FC, `grouped = (1 << 3) - 1 = 7`. This mask includes the expert dimension, so different experts retain their own zero points. The observed case worked, but the axis selection and inferred group count were wrong.

#### After the PR, before the fix: correct axis exposes the mask-selection bug

The new orientation-aware helper correctly identifies the IFM-group axis:

```cpp
get_ifm_dim_idx(dzp_layout, true, 3);  // 2
ngroups = dzp_layout.get_dim(2);      // 1
```

However, the per-channel condition still used the non-unit zero-point rank:

```cpp
// dzp_rank is still 2 for [64, N, 1].
if (ngroups == 1 && dzp_rank <= 2) {
    attr->set_zero_points(DNNL_ARG_WEIGHTS, per_oc, {}, dzp_data_type);
}
```

For a 3D FC, `per_oc = 4`: it selects only the output-channel axis in oneDNN's logical weights layout `[expert, K, N]`. Unlike mask 7, it does not select the expert axis. Consequently, oneDNN uses the first expert's zero points for every expert, producing incorrect outputs when zero points differ across experts.

The regression is therefore **not** that the new helper reads the wrong axis. It reads the correct axis, but that changes the branch taken and exposes a separate rank/mask bug.

#### After the fix in commit `8dc71ce8cd3f4b3128d5181d3624d9924a3bdafe`

Mask selection now explicitly checks whether the weights have a batch/expert dimension:

```cpp
const bool has_batch_dim = weight_rank > 2;
if (ngroups == 1 && !has_batch_dim) {
    attr->set_zero_points(DNNL_ARG_WEIGHTS, per_oc, {}, dzp_data_type);
} else {
    attr->set_zero_points(DNNL_ARG_WEIGHTS, grouped, {group_size, 1}, dzp_data_type);
}
```

For `[64, N, K]`, `has_batch_dim` is `true`. Even with `ngroups == 1`, the code keeps the grouped mask 7 and preserves per-expert zero points. Ordinary 2D per-channel weights still use the per-channel branch.

The same batch-dimension check is applied to scale and zero-point attributes during cache loading, keeping grouped handling consistent with fresh compilation. The commit does **not** change IFM-axis calculation, mask construction, or group-size calculation; the zero-point grouped branch continues to use the scale-derived group size.

| Stage | IFM-group axis read | `ngroups` | Zero-point mask | Expert handling |
|---|---|---|---|---|
| Before PR | 1 (incorrect) | N (incorrect) | 7, grouped | Preserved by accident |
| After PR, before fix | 2 (correct) | 1 (correct) | 4, per-channel | Expert axis ignored |
| After fix | 2 (correct) | 1 (correct) | 7, grouped | Expert axis preserved |

**Regression coverage:** 16 oneDNN-forced cases cover 2D and 64-expert weights, both parameter orientations, scale group sizes of 32 and 64, and fresh/cached execution. Zero points vary by expert and output channel, and outputs are checked against analytical expectations. The unit-test target builds successfully, but all 16 cases were skipped locally because the available GPU lacks IMMAD support; numerical validation on supported hardware remains pending.

## 3. Guidance for future PRs

- **Add a cached test variant:** `compressed_int8_transpose_scale_zp_cached`, which calls `test_compressed_int8_transpose_scale_zp(true)`, to cover the oneDNN `save()`/`load()` path.
- **Add oneDNN (immad) coverage for:**
  - Non-transposed per-channel `[1, N]` scale/ZP.
  - 1D `[N]` scale.
  - Grouped `[G, N]` scale/ZP.
- **Rank assumption:** `get_ifm_dim_idx` assumes the scale/ZP has the same number of real dimensions as the weights. Revisit it if scale/ZP can be broadcast to a different rank.
- **Make sure `create()` and `load()` use the same rank source:** `create()` uses `weight_rank` (the non-unit count), while `load()` uses the serialized `weights_rank`.
- **Horizontal fusion:**
  - It still requires 2D weights.
  - Scalar ZPs are not concatenated, so they don't depend on orientation.
- **New transformations that rebuild a `FullyConnectedCompressed`** must forward `get_transpose_b_scale()` and `get_transpose_b_zp()`. The default `true` silently brings back the old layout assumption.
- **Attribute checks in tests:** whenever an expected model sets non-default FC attributes, enable `ATTRIBUTES` in that test.
- **Before pushing:**
  - Run clang-format (the `ov-ensure-coding-style` skill) on the touched files.
  - Run the `ConvertFCToCompressed*`, `FullyConnectedHorizontalFusion*` and `fully_connected_gpu_tests.compressed_int8_transpose_scale_zp*` tests, both on a device with immad (oneDNN) and on one without (OCL).
