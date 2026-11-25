# Layer split vs tensor split: two-GPU decode profiling

## Setup

- Hardware: 2x Intel Arc B70 (BMG G31, 32 GiB), Linux, Mesa 26.2.4 (Xe kernel driver). [FM-RIL-GNR-G31-006](https://gtax-ril-fm.intel.com/#/clients/18529?tab=jobs)
- Model: `MAI-Code-1.1-Flash-Q2-XS.gguf` (52.44 GiB), fully offloaded (`-ngl 999`).
- Build: llama.cpp Vulkan (`build-vk-sdk/bin/llama-server`).
- Server flags: `-c 131072 -fa on -np 1 --no-host --fit off -ctk f16 -ctv f16 -b 2048 -ub 1024 --ctx-checkpoints 0 --no-cache-prompt --spec-type none`, with `-sm layer` or `-sm tensor`.
- DFlash: off (`--spec-type none`). `-fa on` is flash attention, not DFlash.
- Request: deterministic 8,192-token synthetic prompt, 64 output tokens, `temperature=0`, `ignore_eos=true`.

## Headline numbers

| Mode | Prefill tok/s | Decode tok/s | Decode ms/token |
|---|---:|---:|---:|
| Layer split | 1673 | 64.6 | 15.5 |
| Tensor split | 1152 | 30.6 | 32.7 |
| Tensor split, all-reduce skipped (wrong output, profiling only) | 1773 | 76.7 | 13.0 |

## How each mode moves data between GPUs

- **Layer split:** each GPU holds whole layers. Per decode token there is one handoff: GPU0 sends its output activation (14 KiB) to GPU1.
- **Tensor split:** each layer's weight matrices are divided between the GPUs. Each GPU computes a partial result, and an **all-reduce** sums the partials so both GPUs hold the full result before the next step. This happens **84 times per decode token**, about twice per layer: after attention and after the feed-forward/MoE block. Each all-reduce carries 12–28 KiB.

## Method 1: scheduler handoff timing (layer split)

Gated by `GGML_SCHED_PROFILE=1` in `ggml_backend_sched_compute_splits` / `ggml_backend_sched_copy_input` (`ggml/src/ggml-backend.cpp`). Host `ggml_time_us()` timers wrap each GPU-to-GPU input copy:

- `cross_gpu_source_wait_us`: `ggml_backend_synchronize(input_backend)`, waiting for the source GPU's queued work.
- `cross_gpu_dest_wait_us`: destination event/backend waits.
- `cross_gpu_transfer_us`: async-copy attempt plus fallback `ggml_backend_tensor_copy()` (API wall time, not hardware transfer time).

One `SCHED_PROFILE` line is emitted per graph. The 2 warm-up and 8 prompt graphs (1,024 tokens each) were excluded, leaving 63 decode graphs. The first output token is produced by prompt evaluation.

| Layer-split decode (63 handoffs, 14,336 B each, all synchronous) | Total | Per handoff |
|---|---:|---:|
| Whole handoff call | 338.6 ms | 5.37 ms |
| Source GPU wait | 312.7 ms | 4.96 ms |
| Destination wait + transfer path | 25.8 ms | 0.41 ms |

The source wait mostly overlaps useful GPU0 compute, so it is not overhead. The real handoff cost (destination wait plus transfer) is about **2.6% of decode time**.

## Method 2: all-reduce submission timing (tensor split)

The scheduler timer reports zero copies for tensor split because its communication goes through Vulkan's all-reduce (`ggml_backend_vk_comm_allreduce_tensor` in `ggml/src/ggml-vulkan/ggml-vulkan-comm.cpp`), which runs in `sync_fd` mode on this system. Gated by `GGML_VK_COMM_PROFILE=1`, a host timer around each call logs `VK_COMM_PROFILE`.

| Tensor-split decode | Value |
|---|---:|
| All-reduces | 5,292 (84 per graph × 63 graphs) |
| Host submission time | 178.6 ms total, 2.83 ms per token, 33.7 µs per call |
| Share of decode wall time | 8.7% |

This captures only host-side submission. The GPU-side semaphore waits are queued asynchronously and are not visible to this timer.

## Method 3: counterfactual, all-reduce skipped (tensor split)

`GGML_VK_COMM_SKIP=1` makes the all-reduce return immediately. Output is wrong, but kernel work is unchanged (`ignore_eos` keeps 64 tokens).

- Decode improved from 32.7 to **13.0 ms/token** (30.6 → 76.7 tok/s), faster than layer split.
- The all-reduces therefore cost about **19.7 ms per token (~60% of decode)**, or about **234 µs per all-reduce**.
- Moving 12–28 KiB should take a few µs, so the cost is per-operation sync latency, not bandwidth. In each round trip, both GPUs copy partials to host memory and signal a semaphore, a host thread exchanges sync-fd semaphores, and each GPU waits, reads the peer slot, and adds. The 84 round trips per token are serialized.

## Method 4: GPU busy time per token (both modes)

Source: Intel Xe DRM counters in `/proc/<pid>/fdinfo` (`drm-cycles-ccs` busy, `drm-total-cycles-ccs` at a 19.2 MHz timestamp clock). These counters do not slow execution. The same 8K prompt was sent with 1 and with 129 output tokens; the difference isolates 128 decode tokens. Script: `/tmp/gpu_per_token.py`.

| Mode | Wall ms/token | GPU0 busy ms/token | GPU1 busy ms/token | Total GPU busy ms/token |
|---|---:|---:|---:|---:|
| Layer split | 15.83 | 6.26 | 7.32 | 13.58 |
| Tensor split | 30.73 | 12.53 | 12.77 | 25.30 |

- **Layer split:** the GPUs run one after the other. 13.6 of 15.8 ms per token is GPU work; ~2.2 ms is handoff plus host overhead.
- **Tensor split:** each GPU is busy ~12.6 ms per token, about the same as both GPUs combined in layer split. Total GPU work is ~1.9× layer split. A likely reason, not measured directly: many small decode kernels (norms, routing, small matrix-vector ops) cost about the same on half the data as on all of it. Each GPU is then idle ~18 ms per token, waiting on all-reduce synchronization.

Caveat: the busy counter is assumed to exclude time a GPU context spends blocked on a peer semaphore. The ~40% busy figures are consistent with that, but it is not confirmed from the driver.

## Where tensor split's decode time goes

| Component (per token) | ms |
|---|---:|
| Each GPU computing (in parallel) | ~12.6 |
| Waiting on all-reduce synchronization | ~18.1 |
| **Total wall** | **30.7** |

## Conclusions

1. Tensor split's slowdown is mainly **all-reduce synchronization latency**: ~234 µs × 84 per token, about 60% of decode time. Bandwidth is not the limit.
2. The second problem is **work inflation**: each tensor-split GPU spends about as long per token as both GPUs combined in layer split. Even with free communication, tensor split would reach only ~12.6–13.0 ms/token versus 15.8 for layer split.
3. Layer split's single handoff costs only ~0.4 ms per token of true transfer and wait. Its ~2.2 ms gap is small next to the GPU work.
4. Possible improvements for tensor split: fewer reductions per token (fusing or batching), a lower-latency path (direct peer-to-peer GPU memory instead of host-memory staging with semaphore hand-offs), and reducing replicated or fixed-cost work per GPU.

## Tooling notes

- `GGML_VK_PERF_LOGGER=1` serializes every kernel (tensor decode fell to 3.1 tok/s), so it cannot show cross-GPU wait. In layer mode it aborts with `GGML_ASSERT(ctx->compute_ctx.expired())` at `ggml-vulkan.cpp:14598`.
- A system HTTP proxy intercepts localhost requests (HTTP 403); benchmark clients must bypass it (`ProxyHandler({})`).

## [profiling patches](https://github.com/Kotomi-Du/llama.cpp/tree/profiling)

- `ggml/src/ggml-backend.cpp`: `GGML_SCHED_PROFILE` scheduler handoff timers.
- `ggml/src/ggml-vulkan/ggml-vulkan-comm.cpp`: `GGML_VK_COMM_PROFILE` all-reduce timer, and `GGML_VK_COMM_SKIP` (**produces wrong output**; profiling only).

## Data

| Measurement | Location |
|---|---|
| Layer-split handoff breakdown | `sched-profile-two-gpu-breakdown/` |
| Tensor-split scheduler run (zero copies) | `sched-profile-two-gpu-tensor/` |
| Tensor-split all-reduce timing | `sched-profile-two-gpu-tensor-comm/` |
| Tensor-split all-reduce skipped | `tensor-comm-skip/` |
| Tensor-split per-op GPU profile | `perf-op-tensor/` |
| GPU busy per token | `gpu-per-token-layer.server.log`, `gpu-per-token-tensor.server.log` |

