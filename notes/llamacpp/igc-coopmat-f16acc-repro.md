# IGC reproducer: wrong result (inf) from Vulkan coopmat f16-acc matmul on Windows

## Summary

- One Vulkan compute shader (llama.cpp `matmul_quant_f16_f16acc_aligned_0`, `KHR_cooperative_matrix`, f16 A/B, f16 accumulator) returns `inf` on Windows. CPU reference value is `60344.59` (fits in f16, max 65504).
- Same test, same input, same shader on Linux Mesa (Arc B70, Battlemage G31): passes. Mesa ISA also uses f16 accumulation (`dpas.8x8 ... :hf`).
- Same test on Windows with f32 accumulator (`matmul_quant_f16_aligned_0`): passes.
- Real-world effect: MAI-Code model on llama.cpp gives NaN in layer 40, output is `####...`.

## Environment

| Item | Value |
|---|---|
| GPU | Intel Arc B580 (BMG, `UsDeviceID 0xe20b`), 12 GB |
| OS | Windows 11, build 10.0.26200 |
| Driver | `gfx-driver-ci-master-22006` (IGC `gfx-driver-ci-master-22005`, ReleaseInternal) |
| Vulkan SDK | 1.4.363.1 |
| llama.cpp | https://github.com/Kotomi-Du/llama.cpp.git, branch `win_bug`, commit `95f9555e6` |
| Reference (OK) | Linux, Mesa `26.2.5~kisak1~n`, Arc B70 (Battlemage G31) |

## Test case

`MUL_MAT` in `test-backend-ops`: `C = A x B^T`.

| Tensor | Type | Shape (k x m / k x n) |
|---|---|---|
| A (weights) | Q3_K | 3584 x 3584 |
| B (activations) | F32, uniform [-1000, 1000] | 3584 x 16 |
| C | F32 | 3584 x 16 |

The shape is from the MAI-Code model (`blk.40.ffn_down`). The kernel converts B to f16 and accumulates in f16.

Input data: `mm_q3k_fail.bin` in the llama.cpp repo root, 5,748,736 bytes. It holds raw A bytes (Q3_K blocks, 5,519,360 B) followed by raw B bytes (f32 little-endian, 229,376 B). SHA256: `C1F7E3BB2AFEF9F6EB4E3FAEF4C007E60C8F4BD19360D9C6C4A97C0DD89C559C`.

## Steps

### 1. Build

Prerequisites: Visual Studio 2022 (MSVC), CMake, Vulkan SDK 1.4.363.1.

```powershell
git clone -b win_bug https://github.com/Kotomi-Du/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config RelWithDebInfo --target test-backend-ops -j
```

### 2. Run the failing case

```powershell
$env:GGML_TEST_MM_DATA = (Resolve-Path .\mm_q3k_fail.bin).Path
$env:GGML_VK_PIPELINE_STATS = "matmul"
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max=1000"
```

- `GGML_TEST_MM_DATA`: load A and B from the file (if the file does not exist, the test writes new random data to it).
- `GGML_VK_PIPELINE_STATS`: print the name of each compiled pipeline that matches.
- `-b Vulkan0`: the B580. Check device order in the log (`ggml_vulkan: 0 = Intel(R) Graphics ...`).

Actual (fails every run):

```
ggml_vulkan: pipeline stats for matmul_quant_f16_f16acc_aligned_0:
[MUL_MAT] inf mismatch: Vulkan0=inf CPU=60344.585938   MUL_MAT(type_a=q3_K,type_b=f32,m=3584,n=16,k=3584,...,b_max=1000.000000): FAIL
```

Expected: `OK` (result matches CPU, NMSE < 5e-4).

### 3. Control runs (all pass)

```powershell
# no coopmat: integer dot product kernel matmul_q3_k_q8_1_0
$env:GGML_VK_DISABLE_COOPMAT = 1
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max=1000"
Remove-Item Env:GGML_VK_DISABLE_COOPMAT
```

```bash
# Linux Mesa, same branch and input file
GGML_TEST_MM_DATA=mm_q3k_fail.bin ./build-vk-sdk/bin/test-backend-ops -b Vulkan0 -o MUL_MAT -p "b_max=1000"
```

```powershell
# f32 accumulator, coopmat still on: kernel matmul_quant_f16_aligned_0
$env:GGML_TEST_MM_F32ACC = 1
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max=1000"
Remove-Item Env:GGML_TEST_MM_F32ACC
```

| Platform | Kernel | Accumulator | Result |
|---|---|---|---|
| Windows | `matmul_quant_f16_f16acc_aligned_0` (coopmat) | f16 | FAIL `inf` |
| Windows | `matmul_quant_f16_aligned_0` (coopmat) | f32 | OK |
| Windows | `matmul_q3_k_q8_1_0` (no coopmat) | int32/f32 | OK |
| Linux Mesa | `matmul_quant_f16_f16acc_aligned_0` (coopmat) | f16 (`dpas.8x8 :hf`) | OK |

## Shader dumps

Windows: [igc-dump/](https://github.com/Kotomi-Du/llama.cpp/tree/win_bug/igc-dump) in the branch (`IGC_ShaderDumpEnable=1`, `IGC_DumpToCurrentDir=1`, run step 2 from an empty folder):

- `igc-dump/SM_asm32dcea3890aabdfc.spv` / `.spvasm`: the failing kernel. Coopmat types are `OpTypeCooperativeMatrixKHR %half ...`. The f16 accumulate is `OpCooperativeMatrixMulAddKHR` at `.spvasm` line 1646. The f16 -> f32 convert is `OpFConvert` at line 1809.
- No `.asm` / `.visaasm` was produced, even with these registry values under `HKLM\SOFTWARE\INTEL\IGFX\IGC`: `ShaderDumpEnable=1`, `DumpToCustomDir`, `ShaderDumpEnableAll=1`, `EnableVISADumpCommonISA=1`. `ProcessInfo.txt` shows that IGC read them.

Linux Mesa: [mesa_dump/mai-mesa-matmul.isa](https://github.com/Kotomi-Du/llama.cpp/blob/win_bug/mesa_dump/mai-mesa-matmul.isa) in the branch, SIMD32, 3076 instructions, 16x `dpas.8x8 (16) rN:hf rN:hf rN:hf rN:hf`, then `mov rN:f rM:hf` and 32-bit store.

## Possible second issue: empty GEN disassembly

`vkGetPipelineExecutableInternalRepresentationsKHR` (pipeline made with `VK_PIPELINE_CREATE_CAPTURE_INTERNAL_REPRESENTATIONS_BIT_KHR`) reports one entry `'Gen Disassembly' (Final GEN disassembly for the generated shader binary.)`, `dataSize` 351,569. The fill call returns `VK_SUCCESS`, but all bytes are 0. Repro:

```powershell
$env:GGML_TEST_MM_DATA = (Resolve-Path .\mm_q3k_fail.bin).Path
$env:GGML_VK_PIPELINE_IR = "matmul_quant_f16_f16acc_aligned"
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max=1000"
# writes matmul_quant_f16_f16acc_aligned_0.0.txt in the current folder
```

## Asks for IGC team

1. Why does the f16-acc coopmat kernel give `inf` on Windows when Mesa gives the right result for the same SPIR-V and input?
2. How to get native ISA (`.asm` / `.visaasm`) for this kernel on this driver build?
3. Is the all-zero `Gen Disassembly` from `VK_KHR_pipeline_executable_properties` a known issue?
