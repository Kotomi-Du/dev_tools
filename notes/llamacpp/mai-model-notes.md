# llama.cpp + MAI-Code-1.1-Flash (Vulkan)

## 1. MAI model summary

Read from GGUF header (`C:\Users\yarudu\mai-code\gguf_info.py`).

### Memory needs

| Context | Weights | KV cache (f16 default) | KV cache (q8) | Total RAM (f16) | Total RAM (q8) |
|---|---|---|---|---|---|
| 128K | 56.3 GB | 3.57 GB | 1.79 GB | ~62-66 GB | ~60-64 GB |
| 256K | 56.3 GB | 7.07 GB | 3.54 GB | ~65-70 GB | ~62-67 GB |

### Architecture (`maion-coder`, "Microsoft Aion Flash 1.0", 512x4.9B)

| Item | Value |
|---|---|
| Layers | 42 |
| Hidden size | 3584 |
| Attention | GQA, 48 Q heads / 8 KV heads, head dim 128, QK RMSNorm, sandwich norms |
| RoPE | base 10000, dim 128 |
| Attention pattern | 5 SWA (window 512) : 1 full; full layers = 5, 11, 17, 23, 29, 35, 41 |
| FFN | Even layers dense (FFN 3584); odd layers MoE |
| MoE | 512 experts, top-8, expert FFN 2688, latent dim 1536 (3584 -> 1536 -> experts -> 3584) |
| Router | F32 |
| Vocab | 201,088 (GPT-4o / o200k BPE) |
| Max context | 262,144 |
| Default sampling | temp 0.1, top_p 0.95, top_k 4, min_p 0.3 |

### Parameters

| | Params | Size |
|---|---|---|
| Total | ~137.9B | 56.3 GB |
| Active per token | ~6.0B (~6.8B with embeddings) | ~2.8 GB read |

Experts are ~97% of the file (~54.1 GB).

### Per-layer format

Attention block (all 42 layers): 51.4M params, ~20.0 MB

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `attn_q` | 3584 x 6144 | Q2_K | 7,225,344 |
| `attn_k` | 3584 x 1024 | Q2_K | 1,204,224 |
| `attn_v` | 3584 x 1024 | Q4_K | 2,064,384 |
| `attn_output` | 6144 x 3584 | Q3_K | 9,461,760 |
| norms | 128 / 3584 | F32 | small |

Dense layer (0, 2, ..., 40): 89.9M params, ~33.9 MB (all active)

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `ffn_gate` | 3584 x 3584 | Q2_K | 4,214,784 |
| `ffn_up` | 3584 x 3584 | Q2_K | 4,214,784 |
| `ffn_down` | 3584 x 3584 | Q3_K | 5,519,360 |

MoE layer (1, 3, ..., 41): 6.41B params, ~2.61 GB; active 163.3M (~71 MB)

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `ffn_gate_inp` (router) | 3584 x 512 | F32 | 7,340,032 |
| `ffn_latent_down` | 3584 x 1536 | Q2_K | 1,806,336 |
| `ffn_latent_up` | 1536 x 3584 | Q2_K | 1,806,336 |
| `ffn_gate_exps` | 1536 x 2688 x 512 | Q2_K | 693,633,024 |
| `ffn_up_exps` | 1536 x 2688 x 512 | Q2_K | 693,633,024 |
| `ffn_down_exps` | 2688 x 1536 x 512 | Q4_0 | 1,189,085,184 |

Global:

| Tensor | Shape | Type | Bytes |
|---|---|---|---|
| `token_embd` | 3584 x 201088 | Q2_K | 236,479,488 |
| `output` | 3584 x 201088 | Q6_K | 591,198,720 |

Tensor count by type: Q2_K 211, Q3_K 63, Q4_K 42, Q4_0 21, Q6_K 1, F32 274.

### Quantization

| Type | BPW | Symmetric? |
|---|---|---|
| Q2_K | 2.625 | Asymmetric (scale + min) |
| Q3_K | 3.44 | Symmetric |
| Q4_K | 4.5 | Asymmetric (scale + min) |
| Q4_0 | 4.5 | Symmetric |
| Q6_K | 6.56 | Symmetric |

Original training precision is not recorded in the GGUF.

### KV cache (F16 default)

Per token per layer: 8 x 128 x 2 (K+V) x 2 B = 4 KiB.

| Context | Full layers (7) | SWA layers (35, ~2560 cells) | Total |
|---|---|---|---|
| 262144 | 7.0 GiB | 350 MiB | ~7.3 GiB |
| 131072 | 3.5 GiB | 350 MiB | ~3.9 GiB (matches log) |
| 32768 | 0.9 GiB | 350 MiB | ~1.2 GiB |

Use `-ctk q8_0 -ctv q8_0` to roughly halve it.

### Draft model (`dflash-v3-lc1.gguf`, ~650M)

- DFlash block-diffusion drafter, 5 layers, non-causal attention, block size 8.
- Reads target hidden states from layers 2, 11, 21, 31, 40.
- Weights Q4_K (+4 Q6_K), F32 norms.
- KV at 131072 ctx: 512 MiB (1 full layer) + 160 MiB (4 SWA layers, window 8192).

## 2. Windows setup

### 2.1 Download the MAI model

Source: Azure ML registry (MIT license), asset `azureml://registries/azureml/models/MAI-Code-1.1-Flash-On-Device/versions/1`.

Save as `C:\Users\yarudu\mai-code\download.ps1`:

```powershell
# Never print the SAS URL: it is a time-limited credential.
$ErrorActionPreference = 'Stop'
$root = Join-Path $HOME 'mai-code'
New-Item -ItemType Directory -Force -Path "$root\tools", "$root\models" | Out-Null
Set-Location $root

$azcopy = "$root\tools\azcopy.exe"
if (-not (Test-Path $azcopy)) {
    $zip = "$env:TEMP\azcopy.zip"
    Invoke-WebRequest -Uri 'https://aka.ms/downloadazcopy-v10-windows' -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath "$env:TEMP\azcopy" -Force
    Copy-Item (Get-ChildItem "$env:TEMP\azcopy\azcopy_windows_amd64_*\azcopy.exe" | Select-Object -First 1).FullName $azcopy
}

$ep = 'https://api.catalog.azureml.ms/modelregistry/v1.0/registry/models/nonazureaccount?assetId=azureml://registries/azureml/models/MAI-Code-1.1-Flash-On-Device/versions/1'
$sas = (Invoke-RestMethod -Uri $ep).blobSasUri
if (-not $sas) { throw 'no SAS in registry response' }

$base, $query = $sas -split '\?', 2
$src = "$base/5/*?$query"
& $azcopy copy $src "$root\models\" --output-level essential --log-level ERROR
if ($LASTEXITCODE -ne 0) { throw "azcopy failed ($LASTEXITCODE)" }

Remove-Variable sas, src, base, query
Get-ChildItem "$root\models" | Format-Table Name, Length
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File $HOME\mai-code\download.ps1
```

Expected in `C:\Users\yarudu\mai-code\models` (~57 GB, ~8.5 min):

| File | Size (bytes) |
|---|---|
| `MAI-Code-1.1-Flash-Q2-XS.gguf` | 56,307,620,832 |
| `dflash-v3-lc1.gguf` (draft model) | 381,214,016 |
| `llama-config.json` | 504 |

### 2.2 Build llama.cpp (Vulkan, w64devkit / MinGW)

Vulkan SDK: `C:\VulkanSDK\1.4.363.1`. w64devkit: `C:\w64devkit`. One-time Vulkan setup, run inside `C:\w64devkit\w64devkit.exe`:

```sh
SDK_VERSION=1.4.363.1
VULKAN_SDK=C:/VulkanSDK/$SDK_VERSION
W64DEVKIT_HOME=C:/w64devkit

cp "$VULKAN_SDK/Bin/glslc.exe" "$W64DEVKIT_HOME/bin/"
cp "$VULKAN_SDK/Lib/vulkan-1.lib" "$W64DEVKIT_HOME/lib/"
cp -r "$VULKAN_SDK/Include/"* "$W64DEVKIT_HOME/include/"
mkdir -p "$W64DEVKIT_HOME/lib/pkgconfig"
cat > "$W64DEVKIT_HOME/lib/pkgconfig/vulkan.pc" <<EOF
Name: Vulkan-Loader
Description: Vulkan Loader
Version: $SDK_VERSION
Libs: -lvulkan-1
EOF
```

Note: new w64devkit has no `x86_64-w64-mingw32/` folder; use `$W64DEVKIT_HOME/lib` and `$W64DEVKIT_HOME/include`.

Build (needs CMake in PATH):

```sh
cd C:/Users/yarudu/Documents/project/llama.cpp
cmake -B build -G "MinGW Makefiles" -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Check devices:

```powershell
.\build\bin\RelWithDebInfo\llama-cli.exe --list-devices
```

### 2.3 Run commands

Local dev configuration:

| Item | Value |
|---|---|
| RAM | 128 GB |
| CPU | Intel Alder Lake (ADL) |
| iGPU | ADL iGPU (Vulkan1) |
| dGPU | Intel Arc B580, 12 GB (Vulkan0) |

Common variables:

```powershell
$M = "$HOME\mai-code\models"
$B = "C:\Users\yarudu\Documents\project\llama.cpp\build\bin\RelWithDebInfo"
```

Use `llama-server.exe` for HTTP (web UI at http://127.0.0.1:8080, OpenAI API at `/v1/chat/completions`) or `llama-cli.exe` for terminal chat (same flags, drop `--host`/`--port`).

#### 2.3.1 One GPU: dGPU + experts on CPU (local dev)

Model (56 GB) does not fit in 12 GB dGPU. `--cpu-moe` keeps expert weights (~54 GB) in RAM on CPU; everything else on dGPU.

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0 -devd Vulkan0 -ngl 999 --cpu-moe `
  -c 131072 -np 1 -fa on -b 2048 -ub 1024`
  --spec-type draft-dflash --spec-draft-ngl all --spec-draft-n-max 2 --spec-draft-p-min 0.2 `
  --ctx-checkpoints 0 -fit off -lv 4
```

Result: No output, functional issue. `$env:GGML_VK_DISABLE_COOPMAT = 1` fixed the issue, perf 7TPS. 

##### Finding: wrong output from Vulkan coopmat matmul on dGPU

Summary:

- Symptom: web UI shows nothing; server keeps generating (16k+ tokens). Raw output is one token repeated: `########...`.
- Trigger: first prompt batch with more than 8 tokens. After that, every request on the same server gives `####` (bad values stay in KV cache). Restart server to recover.
- Cause: coopmat (cooperative matrix) matmul path on dGPU (`Intel(R) Graphics`, driver `gfx-driver-ci-master-22006`). Vulkan uses mat-vec kernels for <= 8 columns and matmul kernels above that.
- Not the cause: DFlash draft, flash attention, op offload, prompt cache, context size.
- `test-backend-ops -b Vulkan0` passes all ops (`MUL_MAT`, `MUL_MAT_ID`, `FLASH_ATTN_EXT`, ...), so the bug shows only with real model shapes/types.
- Workaround: `$env:GGML_VK_DISABLE_COOPMAT = 1` before starting the server (or `-ub 8`, very slow prompt).

| Server flags (fresh server each row) | 3-token prompt | 18-token prompt | 3-token prompt after |
|---|---|---|---|
| default (`-ub 1024`) | OK | `####` | `####` |
| no draft model | OK | `####` | `####` |
| `-fa off` | OK | `####` | `####` |
| `--no-op-offload` | OK | `####` | `####` |
| `cache_prompt` true/false | OK | `####` | `####` |
| `-ub 16` | OK | `####` | `####` |
| `-ub 8` | OK | OK | OK |
| `GGML_VK_DISABLE_COOPMAT=1`, `-ub 16` | OK | OK | OK |

How to find the issue:

1. Start the server on a fixed port (window 1), same flags as above plus `--port 8090`.
2. Use the API, not the web UI. The web UI hides the thinking block, and `/v1/chat/completions` returns `500 ... does not match the expected peg-native format` when output is garbage.
3. Send raw prompts to `/completion` (no output parser) and check text and timings (window 2):

```powershell
function Test-Prompt($name, $prompt) {
    $body = @{ prompt = $prompt; n_predict = 40; cache_prompt = $false } | ConvertTo-Json
    $r = Invoke-RestMethod http://127.0.0.1:8090/completion -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 900
    "{0,-10} prompt_n={1,-4} stop={2,-6} out=[{3}]" -f $name, $r.timings.prompt_n, $r.stop_type, ($r.content -replace "`n", ' ')
}
$words = ('The quick brown fox jumps over the lazy dog. ' * 40)
Test-Prompt 'A short-1'  'how are you'
Test-Prompt 'B plain-16' ($words.Substring(0, 80))
Test-Prompt 'C short-2'  'how are you'
```

4. Chat prompt as raw text (to see the real model output with `<|channel|>` tags):

```powershell
$tb = @{ messages = @(@{ role = 'user'; content = 'how are you' }) } | ConvertTo-Json -Depth 5
$p  = (Invoke-RestMethod http://127.0.0.1:8090/apply-template -Method Post -Body $tb -ContentType 'application/json').prompt
Test-Prompt 'chat' $p
```

5. Restart the server for each test and change one thing at a time: remove `-md`/`--spec-*`, `-fa off`, `--no-op-offload`, `-ub 8` / `-ub 16`, then env vars `GGML_VK_DISABLE_COOPMAT`, `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT`, `GGML_VK_DISABLE_FUSION`.
6. Check kernels alone: `test-backend-ops.exe -b Vulkan0 -o MUL_MAT` (also `MUL_MAT_ID`, `FLASH_ATTN_EXT`, `ROPE`, `RMS_NORM`, ...).

7. Find the first NaN tensor: dump every graph node with `llama-eval-callback` (16-token prompt, so the matmul path is used), then search the log for the first `sum = .*nan`:

```powershell
& "$B\llama-eval-callback.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -dev Vulkan0 -ngl 999 --cpu-moe -c 4096 -b 64 -ub 64 -fa on -fit off `
  -p "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over" -n 1 *> $HOME\mai-code\evalcb_coopmat.log
Select-String -Path $HOME\mai-code\evalcb_coopmat.log -Pattern 'sum = .*nan' | Select-Object -First 1
```

##### Result: first NaN tensor

All layers 0-39 are finite. First NaN is the dense FFN down projection of layer 40:

```
node_2187 = (f32) MUL_MAT(blk.40.ffn_down.weight{3584, 3584, 1, 1}, ffn_swiglu-40{3584, 16, 1, 1}) = {3584, 16, 1, 1}
```

| Item | Value |
|---|---|
| src0 (weight) | `blk.40.ffn_down.weight`, Q3_K, 3584 x 3584 |
| src1 (activation) | `ffn_swiglu-40`, F32, 3584 x 16, outliers up to ~381 |
| dst | F32, 3584 x 16 |
| coopmat ON | `sum = -nan(ind)`; values have f16 rounding (`1229.0000`, `-948.0000`) |
| coopmat OFF | `sum = -2386052.0`, no NaN in whole graph (`1243.9910`, `-939.0754`) |

Cause (likely): on Intel coopmat1, quant matmul uses f16 accumulator (`ggml_vk_get_mul_mat_mat_f16acc()`, `coopmat_acc_f16_support`) and f16 B (`prefer_f16_b` in `ggml_vk_mul_mat_q_f16`). Large layer-40 activations overflow f16 (> 65504) -> inf -> NaN in the next RMS_NORM. `test-backend-ops` passes because random inputs in [-1, 1] never reach f16 range limit.

Small repro: `test_mul_mat_large_b` in `tests/test-backend-ops.cpp` (q3_K x f32, m=3584, n=16, k=3584, src1 uniform in [-b_max, b_max]):

```powershell
cmake --build build --config RelWithDebInfo --target test-backend-ops -j
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max"
$env:GGML_VK_DISABLE_COOPMAT = 1; .\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max"; Remove-Item Env:GGML_VK_DISABLE_COOPMAT
```

| b_max | coopmat ON | coopmat OFF |
|---|---|---|
| 1 | OK | OK |
| 300 | OK | OK |
| 1000 | FAIL: `inf mismatch: Vulkan0=inf CPU=66628.82` | OK |

Confirmed: f16 overflow (CPU value 66628 > f16 max 65504).

Kernel used (from `$env:GGML_VK_PIPELINE_STATS = "m"` + `-p "b_max=1000"`): `matmul_quant_f16_f16acc_aligned_0`, coopmat1 (`KHR_coopmat`), B converted to f16, f16 accumulator. Created by `cm1_create_quant(..., "matmul_quant_f16_f16acc", ...)` in `ggml_vk_load_shaders()` when `coopmat_acc_f16_support`. One run also failed with `Vulkan0=-inf CPU=-51741`, so partial sums inside the kernel overflow even when the final value fits in f16.

Saved input data: set `GGML_TEST_MM_DATA=<file>`. If the file does not exist, the test saves its random `a` (raw Q3_K blocks, 5,519,360 B) and `b` (raw f32, 229,376 B) to it; if it exists, the test loads them. File: `mm_q3k_fail.bin` in the llama.cpp repo root (5,748,736 B), fails every run with `Vulkan0=inf CPU=60344.59`. Raw little-endian bytes, so it works on Windows and Linux x86/ARM.

```powershell
$env:GGML_TEST_MM_DATA = "mm_q3k_fail.bin"
.\build\bin\RelWithDebInfo\test-backend-ops.exe -b Vulkan0 -o MUL_MAT -p "b_max=1000"
```

```bash
GGML_TEST_MM_DATA=mm_q3k_fail.bin ./build-vk-sdk/bin/test-backend-ops -b Vulkan0 -o MUL_MAT -p "b_max=1000"
```

Accumulator check on Windows (saved input, `GGML_VK_PIPELINE_STATS=matmul`; `GGML_TEST_MM_F32ACC=1` sets `GGML_PREC_F32` acc on the test matmul):

| Case | Pipeline | Result |
|---|---|---|
| default | `matmul_quant_f16_f16acc_aligned_0` (coopmat, f16 acc) | FAIL `inf` vs CPU 60344.59 |
| `GGML_TEST_MM_F32ACC=1` | `matmul_quant_f16_aligned_0` (coopmat, f32 acc) | OK |
| `GGML_VK_DISABLE_COOPMAT=1` | `matmul_q3_k_q8_1_0` (int dot, no coopmat) | OK |

Linux Mesa runs the same test OK, also with f16 acc (`dpas.8x8 ... :hf`). So f16 acc alone is not enough; Windows driver f16 acc coopmat path is the suspect.

Possible fixes to try: set `GGML_PREC_F32` acc on this matmul (`ggml_prec_set_acc`), or disable f16acc for Intel coopmat quant matmul.

Next: retest with a newer Intel driver; report to llama.cpp / Intel if it still fails.

#### 2.3.2 One GPU: iGPU only (local dev)

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan1 -devd Vulkan1 -ngl 999 `
  -c 131072 -np 1 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0 -fit off -lv 4
```

Result: functionally work, perf 5TPS.

Note: only ~34 GB lands on the iGPU (29.6 GB weights + 3.9 GB KV + 1.3 GB compute). All 21 `ffn_down_exps` tensors (Q4_0, 1134 MiB each, 23.8 GB total) fall back to CPU (`CPU_REPACK` buffer) because they exceed the iGPU `maxStorageBufferRange` (~1 GiB). See `ggml_backend_vk_device_supports_op()` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. This causes GPU<->CPU transfer in each MoE layer.

#### 2.3.3 Multi GPU: dGPU + iGPU (local dev)

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0,Vulkan1 -sm layer -ts 9,55 -devd Vulkan0 `
  -c 32768 -np 1 -ngl 999 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0
```

Result: functionally work, perf 5TPS.

#### 2.3.4 Multi GPU: dGPU + dGPU ([FM-RIL-GNR-G31-006](https://gtax-ril-fm.intel.com/#/clients/18529))

Machine: Windows, 2x Intel Arc B70 (Battlemage G31), 32 GB each (32,558 MiB reported per adapter). The local `win_bug` build was tested with the MAION_CODER FFN down-matmul f32 accumulator fix enabled by `GGML_TEST_MM_F32ACC=1` (see 2.3.1).

Reproduce with the [PowerShell runner](../../.copilot/session-state/15834ec8-0341-49fe-975a-4c26e8cf1010/files/mai-context-benchmark.ps1) (`-SplitMode layer` or `-SplitMode tensor`). The [benchmark CSV](../../.copilot/session-state/15834ec8-0341-49fe-975a-4c26e8cf1010/files/mai-context-benchmark.csv) contains the launch commands, exact timings, idle and peak memory counters, and startup errors. Layer commands were reconstructed; exact temporary ports were not recorded for every layer configuration.

**Benchmark:** one request per row, one slot, f16 KV, `-b 2048 -ub 1024`, prompt caching disabled, 64 generated tokens. The 128K and 256K values are configured context sizes (131,072 and 262,144 tokens); the longest prompt tested was 8,192 tokens. Tensor split was tested without DFlash only.

**Memory columns:** sampled request peaks in GiB. GPU pairs are Vulkan0 / Vulkan1. Dedicated VRAM is adapter-wide; shared RAM is system memory attributed to the server by Windows per GPU. Host RAM is system-wide, includes shared memory, and was about 14.1 GiB before launch. Do not add these columns together; peaks can occur at different times. `0.00` means rounded to two decimals. GPU engine utilization (%) was not collected.

| Split | Context | Flash Attn | DFlash | Prompt tokens | Prefill tok/s | Decode tok/s | dGPU dedicated VRAM (GiB) | dGPU shared RAM (GiB) | Host RAM (GiB) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| layer | 128K | on | off | 256 | 221 | 75.7 | 30.94 / 28.77 | 0.91 / 0.00 | 67.79 |
| layer | 128K | on | off | 8,192 | 1,930 | 69.5 | 31.03 / 28.86 | 0.92 / 0.01 | 67.96 |
| layer | 128K | on | on | 256 | 162 | 87.1 | 31.43 / 31.42 | 1.74 / 1.40 | 69.67 |
| layer | 128K | on | on | 8,192 | 1,403 | 81.9 | 31.48 / 31.54 | 1.75 / 1.41 | 69.80 |
| layer | 128K | off | off | 256 | 165 | 66.5 | 30.31 / 28.28 | 1.00 / 0.00 | 67.95 |
| layer | 128K | off | off | 8,192 | 847 | 41.6 | 31.37 / 29.70 | 2.51 / 0.00 | 69.62 |
| layer | 128K | off | on | 256 | 139 | 74.4 | 29.87 / 31.57 | 6.31 / 2.71 | 75.45 |
| layer | 128K | off | on | 8,192 | 344 | 39.6 | 31.54 / 31.57 | 6.88 / 5.00 | 78.44 |
| layer | 256K | on | off | 256 | 242 | 64.3 | 31.49 / 31.78 | 2.63 / 0.00 | 69.59 |
| layer | 256K | on | off | 8,192 | 567 | 60.6 | 31.51 / 31.78 | 2.64 / 0.05 | 69.76 |
| layer | 256K | on | on | 256 | 115 | 73.5 | 31.48 / 31.38 | 4.08 / 5.64 | 76.52 |
| layer | 256K | on | on | 8,192 | 1,031 | 61.4 | 31.51 / 31.50 | 3.83 / 5.65 | 76.51 |
| layer | 256K | off | off | 256 | 290 | 50.9 | 30.95 / 30.28 | 1.92 / 0.00 | 68.94 |
| layer | 256K | off | off | 8,192 | 393 | 26.2 | 31.07 / 31.29 | 3.27 / 1.50 | 71.83 |
| layer | 256K | off | on | 256 | 147 | 67.1 | 30.92 / 31.66 | 1.94 / 5.13 | 74.51 |
| layer | 256K | off | on | 8,192 | 270 | 34.8 | 31.08 / 31.66 | 3.28 / 7.09 | 77.95 |
| tensor | 128K | on | off | 256 | 753 | 10.7 | 29.51 / 28.73 | 0.00 / 0.00 | 66.67 |
| tensor | 128K | on | off | 8,192 | 1,524 | 10.4 | 29.58 / 28.78 | 0.01 / 0.01 | 66.78 |
| tensor | 128K | off | off | 256 / 8,192 | Startup failed | N/A | N/A | N/A | N/A |
| tensor | 256K | on | off | 256 | 545 | 6.7 | 31.52 / 30.73 | 0.00 / 0.00 | 66.09 |
| tensor | 256K | on | off | 8,192 | 1,488 | 7.1 | 31.55 / 30.78 | 0.01 / 0.01 | 66.19 |
| tensor | 256K | off | off | 256 / 8,192 | Startup failed | N/A | N/A | N/A | N/A |

- **Layer split:** Flash Attention improved both prefill and decode in every 8K comparison. At 128K, DFlash traded prefill speed (1,930 -> 1,403 tok/s) for faster decode (69.5 -> 81.9 tok/s). At 256K, DFlash improved 8K prefill in this run (567 -> 1,031 tok/s) but barely changed decode (60.6 -> 61.4 tok/s), with higher RAM use. DFlash accepted all 42 draft tokens per synthetic request; gains may differ on real prompts.
- **Tensor split:** Flash Attention off failed during context creation at both context sizes, before either prompt could run. With Flash Attention on, tensor split nearly eliminated shared GPU RAM use, but decode was much slower than layer split. At 256K without DFlash, 8K prefill improved from 567 to 1,488 tok/s while decode fell from 60.6 to 7.1 tok/s.
- **VRAM headroom:** Separate idle checks with layer split, Flash Attention, and DFlash showed 461-487 MiB / 65 MiB free at 128K, and 267 MiB / 40 MiB at 256K. The 256K check reported 8.64 GiB of shared GPU RAM across the two adapters at idle. These are counter-based headroom estimates, not Vulkan allocation guarantees.
- **Host RAM:** With 128K layer split, Flash Attention on, and DFlash off, system RAM reached 67.7 GiB before the first prompt and 68.0 GiB during the 8K request. The 52.44 GiB GGUF is memory-mapped, so resident file-backed weight pages likely account for most of the increase; the counters do not isolate those pages from private or driver allocations. `--no-host` does not prevent model-file mapping or shared GPU memory.

#### Flag reference (from `llama-config.json`)

| Flag | Meaning |
|---|---|
| `-m` / `-md` | Main model / draft model |
| `-c` | Context size (max 262144) |
| `-np 1` | One parallel sequence |
| `-ngl 999` | Offload all layers to GPU |
| `--spec-type draft-dflash` | DFlash block speculative decoding |
| `--spec-draft-n-max 3` | Draft up to 3 tokens per step |
| `--spec-draft-p-min 0.2` | Stop drafting below 0.2 confidence |
| `--spec-draft-ngl all` | Draft model fully on GPU |
| `--flash-attn on` | Flash attention |
| `-b` / `-ub 2048` | Logical / physical batch size |
| `--fit off` | Do not auto-shrink params to fit memory |
| `--ctx-checkpoints 0` | No context checkpoints |
| `--no-host` | No host copy of weights (do not use with `--cpu-moe`) |
| `-dev` / `-devd` | Device(s) for main / draft model |
| `--cpu-moe` / `--n-cpu-moe N` | Keep all / first N layers' experts on CPU |
| `-sm layer` / `-ts a,b` | Split mode / split ratio across GPUs |
| `-lv 4` | Verbose log |

## 3. Linux: 2x Intel Battlemage G31 ([FM-RIL-GNR-G31-006](https://gtax-ril-fm.intel.com/#/clients/18529))

Machine: Linux, 2x Intel Arc B70 (Battlemage G31) dGPU, 32 GB each (`xe` kernel driver) + ASPEED BMC VGA.

Quickly check if mesa driver is available 

```bash
apt-cache policy mesa-vulkan-drivers   # we got 26.2.5~kisak1~n
```

### 3.1 Build tools

```bash
sudo apt-get install -y glslc vulkan-tools ninja-build spirv-headers libcurl4-openssl-dev
```

`git`, `cmake` and a C++ compiler (GCC 13.3) were already installed on our machine. Install them if yours does not have them.

### 3.2 LunarG Vulkan SDK (only for its shader compiler)

Ubuntu 24.04's `glslc` cannot build the integer-dot shaders, which add 19–25% generation speed.

```bash
mkdir -p ~/mai-code/tools/vulkan-sdk && cd ~/mai-code/tools
curl -sSfL https://sdk.lunarg.com/sdk/download/latest/linux/vulkan_sdk.tar.xz -o vulkan_sdk.tar.xz
tar -xJf vulkan_sdk.tar.xz -C vulkan-sdk
ls vulkan-sdk   # we got 1.4.363.0
```

### 3.3 Build

```bash
source /home/gta/mai-code/tools/vulkan-sdk/1.4.363.1/setup-env.sh
cmake -B build-vk-sdk -DGGML_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE=$VULKAN_SDK/bin/glslc
cmake --build build-vk-sdk -j 16
```

### 3.4 Check GPU usage

`sudo intel_gpu_top` fails with `no discrete/integrated i915 devices found`: Battlemage uses the `xe` driver, older `intel_gpu_top` supports only i915. `/sys/class/drm/card*/device/gpu_busy_percent` is also empty.

Options:

```bash
lspci -k | grep -A3 -iE 'vga|display'      # confirm driver (xe)
sudo apt install nvtop && nvtop            # nvtop >= 3.1 supports xe (or: sudo snap install nvtop)
cargo install --locked qmassa && sudo ~/.cargo/bin/qmassa   # Intel xe monitor
# no tools: per-process DRM stats (drm-cycles-* rise when GPU is busy)
watch -n1 "sudo cat /proc/$(pgrep -f llama-server)/fdinfo/* 2>/dev/null | grep -E '^drm-(cycles|total-cycles|total|resident)'"
```

igt-gpu-tools >= 2.0 `intel_gpu_top` also supports xe.

High CPU (~97%) on `llama-server` with `-ngl 999` is normal: the main thread spins while waiting for the GPU. Confirm offload in the log (`offloaded X/X layers to GPU`).

### 3.5 Run commands

Layer split (works, with DFlash draft):

```bash
export B=~/mai-code/llama.cpp/build-vk-sdk/bin
export M=~/mai-code/models

$B/llama-server -m $M -ngl 999 -fa on -np 1 --no-host --fit off -sm layer -c 131072 -ctk f16 -ctv f16 -b 2048 -ub 1024 \
  -md models/dflash-v3-lc1.gguf --spec-type draft-dflash --spec-draft-ngl all --spec-draft-n-max 2 --spec-draft-p-min 0.2 --ctx-checkpoints 0
```

Tensor parallel (`-sm tensor`) without draft model:

```bash
$B/llama-server -m $M -ngl 999 -fa on -np 1 --no-host --fit off -sm tensor -c 131072 -ctk f16 -ctv f16 -b 2048 -ub 1024 --ctx-checkpoints 0
```

`-sm` = `--split-mode`, values:

| Value | Meaning |
|---|---|
| `none` | One GPU only |
| `layer` (default) | Split layers and KV across GPUs (pipelined, GPUs take turns) |
| `row` | Split weights by rows (parallel) |
| `tensor` | Split weights and KV across GPUs (parallel, EXPERIMENTAL) |

Optional: `-ts 1,1` for an even split; `GGML_VK_VISIBLE_DEVICES=0,1` to use only the two Vulkan GPUs. `-sm tensor` fails at load with `LLAMA_SPLIT_MODE_TENSOR not implemented for architecture '...'` if the arch is not supported.

### 3.6 Crash: `-sm tensor` + DFlash draft

```
init: DFlash target features via device buffer
ggml/src/ggml-backend-meta.cpp:573: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
... ggml_backend_meta_get_split_state -> ggml_gallocr_alloc_graph -> llama_context::decode -> common_context_can_seq_rm -> server load_model
```

Meaning: in tensor mode a tensor is split along axis 0 (hidden dim) across GPUs. Per-row ops (norm, softmax, get_rows, ...) need full rows on each GPU, so the meta backend asserts. It fires on the first test decode, right after DFlash sets up reading target hidden states from device buffers. Likely cause: DFlash speculative decoding does not support `-sm tensor` yet (a llama.cpp bug, not a command-line error).

### 3.7 benchmark report
#### Setup

- Hardware: two Intel Arc B70 GPUs with Mesa 26.2.4-1ubuntu1~26.04.1; Linux Vulkan build 12307 (36221eedc).
- Server: one slot, f16 KV, `-b 2048 -ub 1024`, `--no-host --fit off --ctx-checkpoints 0 --no-cache-prompt`, following the attached PowerShell runner.
- Requests: 256 and 8192 synthetic prompt tokens per launch, 64 output tokens, `temperature=0`, `seed=42`, `ignore_eos=true`, and `cache_prompt=false`.
- Modes: layer split with and without DFlash; tensor split without DFlash. The table includes successful runs with `GGML_TEST_MM_F32ACC` unset; the summary compares runs with the flag set to `1`.

#### Memory measurements

- GTT GiB: the server process's GPU-accessible system-memory residency from Intel `xe` DRM `drm-resident-gtt` in `/proc/<pid>/fdinfo/*`.
- VRAM GiB: per-process `drm-resident-vram0` from the same files. GPU pairs are in PCI order 0000:17:00.0 / 0000:ba:00.0; these are not Windows adapter-wide counters.
- Host used GiB: system-wide `MemTotal - MemAvailable` from `/proc/meminfo`, not the server's RSS or mapped model size. Reclaimable model-file pages may be resident without increasing this value by their full size; do not add it to GPU residency.
- Sampling: GPU and host peaks were sampled about once per second during each request, not during startup.
- Raw data: [results.jsonl](results.jsonl) retains commands, environment, timings, full completion text, per-request peaks, failures, and server log paths.

| Split | Context | Flash | Draft | Prompt | Prefill tok/s | Decode tok/s | VRAM GiB (PCI pair) | GTT GiB (PCI pair) | Host used GiB |
|---|---:|---|---|---:|---:|---:|---|---|---:|
| layer | 128K | on | off | 256 | 512 | 72.9 | 30.26 / 28.66 | 0.01 / 0.00 | 7.43 |
| layer | 128K | on | off | 8192 | 1691 | 58.4 | 30.29 / 28.70 | 0.02 / 0.00 | 7.52 |
| layer | 128K | on | on | 256 | 263 | 108.0 | 30.80 / 31.75 | 0.01 / 0.01 | 8.42 |
| layer | 128K | on | on | 8192 | 1271 | 92.4 | 30.84 / 31.82 | 0.02 / 0.02 | 8.51 |
| layer | 128K | off | off | 256 | 386 | 69.0 | 29.69 / 28.23 | 0.01 / 0.00 | 7.43 |
| layer | 128K | off | off | 8192 | 1022 | 54.9 | 31.17 / 29.60 | 0.02 / 0.00 | 7.63 |
| layer | 128K | off | on | 256 | 179 | 78.6 | 31.82 / 31.14 | 0.01 / 0.01 | 8.48 |
| layer | 256K | on | off | 256 | 511 | 73.9 | 31.05 / 30.10 | 0.01 / 0.00 | 7.03 |
| layer | 256K | on | off | 8192 | 1684 | 65.0 | 31.08 / 30.13 | 0.02 / 0.00 | 7.10 |
| layer | 256K | off | off | 256 | 513 | 70.4 | 31.20 / 30.23 | 0.01 / 0.00 | 7.50 |
| tensor | 128K | on | off | 256 | 599 | 29.9 | 28.77 / 28.65 | 0.00 / 0.00 | 6.56 |
| tensor | 128K | on | off | 8192 | 1165 | 31.8 | 28.82 / 28.70 | 0.00 / 0.00 | 6.71 |
| tensor | 256K | on | off | 256 | 600 | 31.6 | 30.77 / 30.65 | 0.00 / 0.00 | 6.60 |
| tensor | 256K | on | off | 8192 | 1170 | 31.2 | 30.82 / 30.70 | 0.00 / 0.00 | 6.74 |


#### Summary

- 128K, Flash Attention on, 8K prompt: layer split without DFlash reached 1691 prefill / 58.4 decode tok/s; with DFlash it reached 1271 / 92.4 tok/s; tensor split without DFlash reached 1165 / 31.8 tok/s.
- 256K, Flash Attention on, 8K prompt, no DFlash: layer split reached 1684 prefill / 65.0 decode tok/s versus tensor split at 1170 / 31.2 tok/s.
- F32 accumulator: across 14 configurations where both settings succeeded, `GGML_TEST_MM_F32ACC=1` changed median prefill by +1.0% and median decode by -0.1% versus unset. Pairs varied (prefill -22.7% to +26.6%; decode -6.1% to +29.1%) without a consistent direction or change in success/failure.
- Unsupported configuration: tensor split with Flash Attention off exits during context creation (`SPLIT_MODE_TENSOR requires flash_attn to be enabled`) at 128K and 256K.
- GPU allocation or device loss: layer split at 256K with DFlash failed at startup with either flash mode; layer 128K with Flash Attention off and DFlash failed on the 8K request (`ErrorDeviceLost` after allocation failures); layer 256K with Flash Attention off and no DFlash failed to reserve graph buffers on the 8K request (`ErrorOutOfDeviceMemory`). 

## 4. Summary

Best 256K config per split mode and prompt size on 2x Arc B70 (64GB DRAM), from [section 2.3.4](#234-multi-gpu-dgpu--dgpu-fm-ril-gnr-g31-006) (Windows) and [section 3.7](#37-benchmark-report) (Linux). All use `-c 262144`, f16 KV, `-b 2048 -ub 1024`, 64 output tokens. "Best" = highest decode tok/s; prefill-best is listed when it is a different config.

| OS | `-sm` | Prompt | Best config (FA, DFlash) | Prefill tok/s | Decode tok/s | Prefill-best alternative |
|---|---|---:|---|---:|---:|---|
| Windows | layer | 256 | on, on | 115 | 73.5 | FA off, no DFlash: 290 / 50.9 |
| Windows | layer | 8K | on, on | 1,031 | 61.4 | - |
| Windows | tensor | 256 | on, off | 545 | 6.7 | - |
| Windows | tensor | 8K | on, off | 1,488 | 7.1 | - |
| Linux | layer | 256 | on, off | 511 | 73.9 | FA off, no DFlash: 513 / 70.4 |
| Linux | layer | 8K | on, off | 1,684 | 65.0 | - |
| Linux | tensor | 256 | on, off | 600 | 31.6 | - |
| Linux | tensor | 8K | on, off | 1,170 | 31.2 | - |

- Windows is not sensitive when DRAM is overflow because shared system RAM can be a backup while linux will fail with `ErrorOutOfDeviceMemory`. The performance meets to the requirement 50TPS. Linux has better performance than windows.
- Flash Attention improves prefill and decode, and reduces memory for long prompts (8K); for short prompts memory is about the same. DFlash improves decode but usually lowers prefill and always uses more memory (at 256K on Linux it does not fit).
- Tensor split: needs Flash Attention on, does not work with DFlash (see 3.6), and decodes far slower than layer split (Linux ~31 tok/s, Windows ~7 tok/s). It gives the best Windows 8K prefill (1,488 tok/s) and near-zero shared GPU RAM / GTT.
- Linux layer split, Flash Attention off, 8K prompt fails with `ErrorOutOfDeviceMemory`; keep Flash Attention on at 256K.
- 128K is faster than 256K on windows, not the case for linux. One possible cause of the Windows gap: at 256K, Windows uses more shared system RAM (about 2-5 GiB on GPU0 vs about 1 GiB at 128K). The VRAM is close to full, so some memory traffic probably goes over PCIe. On Linux, GTT stays near 0 at both sizes.
- Does two-dGPU configuration improve the performance? yes, because CPU moe is too slow. At 128K with an 8K prompt and DFlash on, layer split reached 92.4 tok/s versus 53.2 tok/s on one dGPU, about 1.7× faster.
- Profiling notes: [Layer split vs tensor split: two-GPU decode profiling](layer-vs-tensor-split-profiling.md)

## Next Step
1. use MSFT CGC API
2. profiling computation bottleneck for tensor split
3. try oneccl for peer to peer GPU communication
4. KV cache int8
5. create taskML to report benchmark data with customized binary

