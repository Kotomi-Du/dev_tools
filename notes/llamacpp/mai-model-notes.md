# llama.cpp + MAI-Code-1.1-Flash (Windows, Vulkan)

Machine: i9-13900K, 128 GB RAM, Vulkan0 = Intel dGPU (12 GB), Vulkan1 = Intel UHD 770 iGPU (shared RAM, ~64 GB).

## 1. Build llama.cpp for Vulkan backend

Vulkan SDK: `C:\VulkanSDK\1.4.363.1`

### build with w64devkit (MinGW)

w64devkit: `C:\w64devkit`. One-time Vulkan setup, run inside `C:\w64devkit\w64devkit.exe`:

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

### Check devices

```powershell
.\build\bin\RelWithDebInfo\llama-cli.exe --list-devices
```

## 2. Download the MAI model

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

## 3. MAI model summary

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

## 4. Run commands

Common variables:

```powershell
$M = "$HOME\mai-code\models"
$B = "C:\Users\yarudu\Documents\project\llama.cpp\build\bin\RelWithDebInfo"
```

Use `llama-server.exe` for HTTP (web UI at http://127.0.0.1:8080, OpenAI API at `/v1/chat/completions`) or `llama-cli.exe` for terminal chat (same flags, drop `--host`/`--port`).

### 4.1 One GPU: dGPU + experts on CPU (local dev)

Model (56 GB) does not fit in 12 GB dGPU. `--cpu-moe` keeps expert weights (~54 GB) in RAM on CPU; everything else on dGPU.

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0 -devd Vulkan0 -ngl 999 --cpu-moe `
  -c 131072 -np 1 -fa on -b 2048 -ub 1024`
  --spec-type draft-dflash --spec-draft-ngl all --spec-draft-n-max 2 --spec-draft-p-min 0.2 `
  --ctx-checkpoints 0 -fit off -lv 4
```

Result: No output, functional issue. `$env:GGML_VK_DISABLE_COOPMAT = 1` fixed the issue, perf 7TPS. 

#### Finding: wrong output from Vulkan coopmat matmul on dGPU

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

Next: find which matmul (weight type + shape) gives NaN, for a small repro; retest with a newer Intel driver; report to llama.cpp / Intel if it still fails.

### 4.2 One GPU: iGPU only (local dev)

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan1 -devd Vulkan1 -ngl 999 `
  -c 131072 -np 1 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0 -fit off -lv 4
```

Result: functionally work, perf 5TPS.

Note: only ~34 GB lands on the iGPU (29.6 GB weights + 3.9 GB KV + 1.3 GB compute). All 21 `ffn_down_exps` tensors (Q4_0, 1134 MiB each, 23.8 GB total) fall back to CPU (`CPU_REPACK` buffer) because they exceed the iGPU `maxStorageBufferRange` (~1 GiB). See `ggml_backend_vk_device_supports_op()` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. This causes GPU<->CPU transfer in each MoE layer.

### 4.3 Multi GPU: dGPU + iGPU (local dev)

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0,Vulkan1 -sm layer -ts 9,55 -devd Vulkan0 `
  -c 32768 -np 1 -ngl 999 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0
```

Result: functionally work, perf 5TPS.

### 4.4 Multi GPU: dGPU + dGPU ([FM-RIL-GNR-G31-006](https://gtax-ril-fm.intel.com/#/clients/18529))
```powershell
$M = "C:\Users\gta\Documents\Models\mai-code\models"
$B = "C:\Users\gta\Documents\project\llama"

 & "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" -ngl 999 -fa on -np 1 --no-host --fit off -sm layer -c 131072 -ctk f16 -ctv f16 -b 2048 -ub 1024 `
 --spec-type draft-dflash --spec-draft-ngl all --spec-draft-n-max 2 --spec-draft-p-min 0.2 --ctx-checkpoints 0
```
Result: doesn't work, because of section 3.1 issue.

### Flag reference (from `llama-config.json`)

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



- DFlash block-diffusion drafter, 5 layers, non-causal attention, block size 8.
- Reads target hidden states from layers 2, 11, 21, 31, 40.
- Weights Q4_K (+4 Q6_K), F32 norms.
- KV at 131072 ctx: 512 MiB (1 full layer) + 160 MiB (4 SWA layers, window 8192).

## 5. Linux: 2x Intel Battlemage G31 ([FM-RIL-GNR-G31-006](https://gtax-ril-fm.intel.com/#/clients/18529))

Machine: Linux, 2x Intel Battlemage G31 dGPU (`xe` kernel driver) + ASPEED BMC VGA.

Quickly check if mesa driver is available 

```bash
apt-cache policy mesa-vulkan-drivers   # we got 26.2.5~kisak1~n
```

### 5.1 Build tools

```bash
sudo apt-get install -y glslc vulkan-tools ninja-build spirv-headers libcurl4-openssl-dev
```

`git`, `cmake` and a C++ compiler (GCC 13.3) were already installed on our machine. Install them if yours does not have them.

### 5.2 LunarG Vulkan SDK (only for its shader compiler)

Ubuntu 24.04's `glslc` cannot build the integer-dot shaders, which add 19–25% generation speed.

```bash
mkdir -p ~/mai-code/tools/vulkan-sdk && cd ~/mai-code/tools
curl -sSfL https://sdk.lunarg.com/sdk/download/latest/linux/vulkan_sdk.tar.xz -o vulkan_sdk.tar.xz
tar -xJf vulkan_sdk.tar.xz -C vulkan-sdk
ls vulkan-sdk   # we got 1.4.363.0
```

### 5.3 Build

```bash
source /home/gta/mai-code/tools/vulkan-sdk/1.4.363.1/setup-env.sh
cmake -B build-vk-sdk -DGGML_VULKAN=ON -DVulkan_GLSLC_EXECUTABLE=$VULKAN_SDK/bin/glslc
cmake --build build-vk-sdk -j 16
```

### 5.4 Check GPU usage

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

### 5.5 Run commands

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

### 5.6 Crash: `-sm tensor` + DFlash draft

```
init: DFlash target features via device buffer
ggml/src/ggml-backend-meta.cpp:573: GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0) failed
... ggml_backend_meta_get_split_state -> ggml_gallocr_alloc_graph -> llama_context::decode -> common_context_can_seq_rm -> server load_model
```

Meaning: in tensor mode a tensor is split along axis 0 (hidden dim) across GPUs. Per-row ops (norm, softmax, get_rows, ...) need full rows on each GPU, so the meta backend asserts. It fires on the first test decode, right after DFlash sets up reading target hidden states from device buffers. Likely cause: DFlash speculative decoding does not support `-sm tensor` yet (a llama.cpp bug, not a command-line error).

Next steps:

1. Run `-sm tensor` without the draft model (5.5). If it works, the problem is DFlash + tensor split. If it still crashes, the main model graph has an op that tensor mode does not support yet.
2. For a symbolized backtrace: `cmake -B build-vk-dbg -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo && cmake --build build-vk-dbg -j 16`.
3. Workaround: `-sm layer` with DFlash, or `-sm tensor` without the draft model. Compare tokens/s.
4. Search llama.cpp issues for "split-mode tensor" + dflash/speculative; report with the backtrace and a minimal command if none exists.

## 6. Status: 2x dGPU results

| `-sm` | Linux with draft | Linux without draft | Linux GPU usage | Windows with draft | Windows without draft |
|---|---|---|---|---|---|
| `layer` | 74 t/s | 70 t/s | 30% ; 52% | not functional (see 4.1) | |
| `tensor` | failed (see 5.6) | 25 t/s | 54%; 54% | | |
