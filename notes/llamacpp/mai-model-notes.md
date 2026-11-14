# llama.cpp + MAI-Code-1.1-Flash (Windows, Vulkan)

Machine: i9-13900K, 128 GB RAM, Vulkan0 = Intel dGPU (12 GB), Vulkan1 = Intel UHD 770 iGPU (shared RAM, ~64 GB).

## 1. Build llama.cpp

Vulkan SDK: `C:\VulkanSDK\1.4.363.1`

### Option A: MSVC (Visual Studio) - used for `build\bin\RelWithDebInfo`

```powershell
cd C:\Users\yarudu\Documents\project\llama.cpp
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config RelWithDebInfo -j
```

Binaries: `build\bin\RelWithDebInfo\` (`llama-server.exe`, `llama-cli.exe`, `llama.exe`, ...).

Note: `llama.exe` needs a subcommand (`llama.exe serve ...` / `llama.exe cli ...`).

### Option B: w64devkit (MinGW)

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

## 3. Run commands

Common variables:

```powershell
$M = "$HOME\mai-code\models"
$B = "C:\Users\yarudu\Documents\project\llama.cpp\build\bin\RelWithDebInfo"
```

Use `llama-server.exe` for HTTP (web UI at http://127.0.0.1:8080, OpenAI API at `/v1/chat/completions`) or `llama-cli.exe` for terminal chat (same flags, drop `--host`/`--port`).

### 3.1 One GPU: dGPU + experts on CPU (recommended)

Model (56 GB) does not fit in 12 GB dGPU. `--cpu-moe` keeps expert weights (~54 GB) in RAM on CPU; everything else on dGPU.

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0 -devd Vulkan0 -ngl 999 --cpu-moe `
  -c 131072 -np 1 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0 -fit off -lv 4
```

Measured: ~12.8 t/s generation.

Tuning: `--n-cpu-moe N` keeps experts of the first N layers on CPU only (each MoE layer's experts ~2.6 GB). Lower `-c` if out of memory.

### 3.2 One GPU: iGPU only

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan1 -devd Vulkan1 -ngl 999 `
  -c 131072 -np 1 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0 -fit off -lv 4
```

Note: only ~34 GB lands on the iGPU (29.6 GB weights + 3.9 GB KV + 1.3 GB compute). All 21 `ffn_down_exps` tensors (Q4_0, 1134 MiB each, 23.8 GB total) fall back to CPU (`CPU_REPACK` buffer) because they exceed the iGPU `maxStorageBufferRange` (~1 GiB). See `ggml_backend_vk_device_supports_op()` in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`. This causes GPU<->CPU transfer in each MoE layer.

### 3.3 Multi GPU: dGPU + iGPU

```powershell
& "$B\llama-server.exe" -m "$M\MAI-Code-1.1-Flash-Q2-XS.gguf" -md "$M\dflash-v3-lc1.gguf" `
  -dev Vulkan0,Vulkan1 -sm layer -ts 9,55 -devd Vulkan0 `
  -c 32768 -np 1 -ngl 999 `
  --spec-type draft-dflash --spec-draft-n-max 3 --spec-draft-p-min 0.2 --spec-draft-ngl all `
  --flash-attn on -b 2048 -ub 2048 --ctx-checkpoints 0
```

- llama.cpp skips the iGPU by default when a dGPU exists, so list both in `-dev`.
- `-sm layer` splits by layer; `-ts 9,55` = ~9 GB on dGPU, rest on iGPU. Lower the first value if dGPU OOM.
- iGPU is slow and uses shared RAM; it usually limits overall speed.

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

## 4. MAI model summary

Read from GGUF header (`C:\Users\yarudu\mai-code\gguf_info.py`).

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
