# [GPU] Support sliced packed sub-byte zero points in FC compression 

Tracking PR: [openvinotoolkit/openvino#39199](https://github.com/openvinotoolkit/openvino/pull/39199)

### Details:
 - **Problem:** ONNX `MatMulNBits` with u2/u4 weights stores zero points packed into bytes. When the number of groups per row doesn't fill whole bytes, the importer pads them (for example, 214 u2 groups take 53.5 bytes, so 216 are stored). It then trims the padding with a `Slice` placed after the `Convert`, because `Slice` can't run on sub-byte types. `SliceToStridedSlice` (part of CommonOptimizations) may later turn that `Slice` into a `StridedSlice`. This slice broke compression in two places:
   1. `MarkDequantization` / `KeepConstPrecision` didn't recognize the zero-point path. `ConstantFolding` then folded `Slice(Convert(u2 ZP))` into an f32/f16 constant, and the low-precision zero point was lost.
   2. The GPU `FC_COMPRESSED_WEIGHT_PATTERN` didn't accept a sliced zero point, so the FC was never turned into `FullyConnectedCompressed`.

 - **Common transformations (`mark_dequantization_subgraph.cpp`):**
   - **New `make_zp_slice_pattern`:** adds an optional `Slice` followed by an optional `StridedSlice` between the zero-point `Convert` and the optional `Reshape`/`Unsqueeze`. Both `MarkDequantization` and `KeepConstPrecision` use it.
   - **Effect:** the zero-point `Convert` is marked with `DisableConstantFolding` and the u2/u4 zero-point `Constant` keeps its precision (`KeepConstPrecision`), so the dequantization subgraph survives `ConstantFolding`. `KeepConstPrecision` runs again after CommonOptimizations, so it must also see the zero point through the `StridedSlice`.
   - **New `is_direct_input` check:** `MarkDequantization` normally swaps the zero-point `Convert` and `Reshape` so the `Reshape` can be folded. The swap now happens only when the `Reshape` consumes the `Convert` directly; moving it across a `Slice`/`StridedSlice` would change the result.

 - **GPU pattern (`compressed_weights_pattern.hpp`):**
   - The zero-point input of `Subtract` can now be `Constant`, `Convert(Constant)`, `Slice(Convert(Constant))` or `StridedSlice(Convert(Constant))`. The slice parameters must be constants.
   - The two `Subtract` alternatives are merged into one `subtract_m` that reuses the same constant and convert sub-patterns.

 - **GPU transformation (`ConvertFullyConnectedToFullyConnectedCompressed`):**
   - For a sliced zero point, the u2/u4 constant is converted to u8, the original slice is applied again, and the result is folded into a u8 `Constant`.
   - From there it follows the existing path unchanged: `reshape_const` once, then `convert_zp_const`.
   - Other zero-point types combined with a slice are rejected, and the graph is left unchanged.

 **Topology example** (N = 32, group size 16, 214 groups used and 216 stored, so K = 3424):

```
Model subgraph (from MatMulNBits):
 Weights u2 [32,214,16]          ZP u2 [32,216,1]   <- padded to whole bytes
        |                               |
 Convert -> f16                  Convert -> f16 [32,216,1]
        |                               |
        |                        (Strided)Slice(axis=1, 0:214) -> [32,214,1]
        |                               |
        +----------- Subtract ----------+
                        |
                    Multiply <-- Scale f16 [32,214,1]
                        |
                    Reshape -> [32,3424]
                        |
 Input [?,3424] --> FullyConnected

Step 1 - MarkDequantization / KeepConstPrecision (common):
 ZP u2 Constant         : KeepConstPrecision     (stays u2)
 ZP Convert             : DisableConstantFolding (not folded with the Slice)
 Subtract / Multiply    : marked as dequantization nodes
 => the graph above survives ConstantFolding unchanged

Step 2 - ConvertFullyConnectedToFullyConnectedCompressed (GPU):
 ZP u2 [32,216,1] -> Convert u8 -> (Strided)Slice -> folded u8 Constant [32,214,1]
                                                  -> reshape_const -> u8 [32,214]

 Input [?,3424] --> FullyConnectedCompressed(
                        weights u2 [32,3424],
                        bias,
                        scale   f16 [32,214],
                        zp      u8  [32,214])
```

 - **Tests:**
   - `MarkDequantizationSlicedZeroPoint`: u2 weights `[32,214,16]`, a u2 zero point `[32,216,1]` trimmed to `[32,214,1]` by a `Slice` on axis 1, and an f32 scale `[32,214,1]`. After `MarkDequantization` → `ConstantFolding` → `KeepConstPrecision`, both u2 constants keep their precision, both `Convert`s are excluded from constant folding, and `Subtract`/`Multiply` are marked as dequantization nodes.
   - `KeepConstPrecisionStridedSlicedZeroPoint`: the same shapes with a `StridedSlice` trimming axis 1 from 216 to 214 groups, representing the subgraph after `SliceToStridedSlice`. `KeepConstPrecision` still reaches both u2 constants and preserves their precision.
   - `ConvertFCToCompressed26_SlicedU2ZeroPoint`: input `[?,3424]`, u2 weights `[32,214,16]`, a u2 zero point `[32,216,1]` trimmed to `[32,214,1]` by a `Slice`, and an f16 scale `[32,214,1]`. The GPU transformation folds `Slice(Convert(ZP, u8))` into a u8 constant `[32,214,1]`, reshapes it once to `[32,214]`, and keeps it unchanged in `convert_zp_const`. The reference `FullyConnectedCompressed` receives weights `[32,3424]`, scale `[32,214]`, and a direct u8 zero-point constant `[32,214]`. Constant-value comparison is enabled.
   - `ConvertFCToCompressed27_StridedSlicedU2ZeroPoint`: the same shapes and expected compressed FC with a `StridedSlice` instead of a `Slice`; constant-value comparison is also enabled.

### Tickets:
 - *ticket-id*

### AI Assistance:
 - AI assistance used: yes
 - AI was used to extend the dequantization patterns, simplify the GPU zero-point pattern, write the sliced zero-point handling in `ConvertFullyConnectedToFullyConnectedCompressed`, and write and update the unit tests. `convert_fc_to_compressed.cpp` and `convert_matmul_to_fc.cpp` were compiled locally (MSBuild, RelWithDebInfo).
 - Validation: `cmake --build build --config RelWithDebInfo --target ov_gpu_unit_tests --parallel 8` passed. `.\bin\intel64\RelWithDebInfo\ov_gpu_unit_tests.exe --gtest_filter=TransformationTestsF.ConvertFCToCompressed*` passed all 27 tests, including both updated sliced-zero-point cases. `git diff --check` passed after the code changes.
 - Low-precision validation is incomplete: `cmake --build build --config RelWithDebInfo --target ov_lp_transformations_tests --parallel 8` was interrupted, so the updated `MarkDequantizationSlicedZeroPoint` and `KeepConstPrecisionStridedSlicedZeroPoint` tests have not been run.
