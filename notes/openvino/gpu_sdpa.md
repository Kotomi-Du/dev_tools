# SDPA Micro Kernel Architecture

## Overview

`sdpa_micro.cl` implements fused Scaled Dot-Product Attention using oneDNN gemmstone micro-kernels for Intel GPUs. `sdpa_gen_micro.cpp` is the host-side code that configures the micro-kernels, generates JIT constants, and dispatches the kernel.

## Kernel Selection: sdpa_micro / sdpa_opt / sdpa_ref

### Level 1: Registry (`registry/sdpa_impls.cpp`)
Order: `SDPAOpt` → `SDPARef`. The first one whose `validate_impl()` returns true is used.
- `SDPAOpt` rejects the node in these cases:
  - Q/K/V/output format is not `bfyx`
  - unsupported precision (K/V may also be `i8`)
  - **dynamic head_size**
  - per-channel KV compression, or asymmetric quantization with planar scale/ZP storage, when micro is not supported
- `SDPARef` checks only format and precision. It is the fallback (`sdpa_ref.cl`).

### Level 2: Inside `SDPAOptImpl` (`sdpa_opt.cpp`)
`supports_micro_sdpa()` returns true only if all of these hold:
- oneDNN is enabled
- `supports_immad` (XMX), arch ≥ xe_hpg, and the microkernel query passes
- not xe3p with head_size ≤ 64
- the Q/K transpose order keeps the last dim in place (V may also swap its last two dims)
- static head counts, with K heads == V heads
- K head_size == V head_size, and both ≤ 512
- the mask is not a non-constant scalar
- 4D output

Runtime dispatch in `execute()`:
1. micro stage exists && prefill && !indirect && (KV not int4 || per-channel KV) → **sdpa_micro prefill**
2. prefill || unaligned head_size → **sdpa_opt multi_tokens** (regular or indirect)
3. micro single-token stage exists && !indirect → **sdpa_micro single token** (this stage is not created on ARL-H)
4. else → **sdpa_opt single_token** (+ finalization if partitions > 1)

Dynamic shapes compile all opt stages, plus the micro stages when micro is supported. If micro kernel creation fails (insufficient registers), prefill falls back to opt.

### Is sdpa_opt used on XMX platforms?
Yes, in these cases:
- indirect KV (beam search)
- `supports_micro_sdpa()` returns false
- micro compilation fails
- int4 KV that is not per-channel (prefill)
- unaligned head_size
- generate on ARL-H

On non-XMX platforms, `sdpa_opt` is the only optimized kernel.

### Sink Support
All three kernels support the sink input via `HAS_SINK_INPUT` / `SINK_DATA_T`. It must be per-head: sink count == Q heads, which `SDPAOpt::validate_impl` enforces (throws otherwise).
- Softmax: `max = max(max, sink)` and `exp_sum += exp(sink - max)`. Sink adds nothing to V.
- opt/ref receive sink as a regular data input. Micro adds the `SINK` argument explicitly.
- Micro with GQA indexes the sink by `b0 * KV_GROUP_SIZE + sg_lid`. With asymmetric quantization and sink present, micro does not fold the zero-point shift into the softmax.
- To check: the PA kernel applies sink only in partition 0, but the single-token sdpa_opt path has no such guard. Not yet confirmed whether finalization compensates; verify with multi-partition decode.

## Two Paths

### Non-PA Path (Standalone SDPA)
- Input: Q `[batch, heads, seq_q, head_size]`, K `[batch, kv_heads, seq_k, head_size]`, V `[batch, kv_heads, seq_k, head_size]`
- Each workgroup handles one (batch, head, query_block) slice
- K/V pointers are offset by `KEY_OFF`/`VAL_OFF` macros using stride defines (`KEY_S2`, `VAL_S2`)
- The k-loop iterates over all key tokens in chunks of `ugemm_kq_wg_tile_m`
- Scale/ZP inputs are separate tensors passed as kernel arguments

### PA Path (Paged Attention)
- **Prefill** (`IS_PREFILL`): K/V are contiguous (same as non-PA layout), dispatched per subsequence
- **Generate** (`!IS_PREFILL`): K/V are in paged blocks accessed via `block_indices[]`
  - Each k-loop iteration looks up the physical block: `block_indices[base_block_index + k_block_num]`
  - Supports `Kc`/`Vc` (current token's K/V from the new input, not yet in cache)
  - `IS_GQA_SINGLE_TOKEN`: batches multiple Q heads into one workgroup for GQA

## Sliding Window

Sliding window (`SLIDING_WINDOW_SIZE > 0`) only applies for PA paths. For non-PA SDPA, `SLIDING_WINDOW_SIZE` is not set.

- **PA generate**: `window_k_begin = max(0, past_len + wg_j0 - SLIDING_WINDOW_SIZE + 1)`
- **PA prefill**: `window_k_begin = max(0, wg_j0 - SLIDING_WINDOW_SIZE + 1)`
- V pointer and V_scales are advanced by `window_k0_begin` to skip tokens outside the window
- The causal mask `greater_than` macro includes the sliding window lower bound check

However, for PA generate (`IS_PAGED_ATTENTION && !IS_PREFILL`), the V sliding window advance is skipped (V is accessed per-block via `block_indices` so the offset is implicit in block selection).

## Connection: sdpa_gen_micro.cpp → sdpa_micro.cl

```
sdpa_gen_micro.cpp
├── init_microkernels()     → selects gemmstone micro-kernels (gemm_kq, gemm_vs)
├── get_jit_constants()     → generates #defines (KEY_GROUP_SIZE, IS_KEY_BY_CHANNEL, etc.)
├── generateShim()          → produces ugemm_kq()/ugemm_vs() function code (JIT string)
├── get_arguments_desc()    → declares kernel argument order (K, Q, V, A, scales, ...)
├── get_dispatch_data_func()→ computes GWS/LWS at runtime
└── get_kernel_data()       → assembles KernelData with .cl source + JIT + micro-kernel packages

sdpa_micro.cl
├── Receives generated ugemm_kq/ugemm_vs as inlined functions via JIT
├── Uses JIT #defines for tile sizes, data types, quantization params
├── Calls ugemm_kq(K, ldk, Q_slm, ..., K_scales, K_zp, ldkq) and ugemm_vs(V, ldv, ...)
└── Implements online softmax (per-tile max tracking, exp, rescaling, normalization by column sums)
```

Key JIT defines from `sdpa_gen_micro.cpp`:
- `ugemm_kq_wg_tile_m/n`, `ugemm_kq_sg_tile_m/n` — tile dimensions
- `KEY_SCALES`, `VAL_SCALES` — quantization mode (0=none, 2=QUANTIZE_2D, 3=QUANTIZE_COMMON)
- `KEY_GROUP_SIZE`, `VAL_GROUP_SIZE` — elements per scale group
- `IS_KEY_BY_CHANNEL`, `IS_VALUE_BY_CHANNEL` — per-channel quantization flags
- `KEY_COMP_S0..S3` — strides for scale tensor offset calculation

## Per-Channel Quantization Support

### Problem
Scale tensor shape `[batch, heads, 1, head_size]` — one scale per channel, broadcast across all tokens. The original code assumed per-token scales (`[batch, heads, seq, 1]`).

### Changes in sdpa_gen_micro.cpp

1. **Detect per-channel from scale shape**:
   ```cpp
   const bool is_key_by_channel = (key_scale_seq == 1 && key_scale_groups > 1);
   ```
   Emits `IS_KEY_BY_CHANNEL` / `IS_VALUE_BY_CHANNEL` JIT defines.

2. **KEY_GROUP_SIZE = 1** (derived from scale shape dim[3] = head_size):
   Each head_size element has its own scale.

3. **aqGroupM = wg_tile_m** (`config->unroll_m_kq * config->wg_m_kq`) for KQ GEMM:
   Covers the entire tile so `m_group = m / aqGroupM = 0` within each tile — scale row never advances.

4. **aqGroupK = 1** for KQ GEMM:
   Each K-dimension (head_size) element gets its own scale column.

5. **VS GEMM reversed**: `aqGroupM = 1` (per head_size row), `aqGroupK = wg_tile_m` (all tokens in tile share).

6. **ldkq = 1** in the kernel for per-channel:
   With Layout::N (column-major: `element(row, col) = ptr[row + col * ld]`), scale access is `ptr[m_group + k_group * ld]` = `ptr[0 + k * 1]` = `ptr[k]`.

### Changes in sdpa_micro.cl

1. **ldkq/ldvq**: Set to 1 when `IS_KEY_BY_CHANNEL` / `IS_VALUE_BY_CHANNEL` (stride between scale columns = 1 element).

2. **Prefetch**: Only prefetches 1 row of scales (not per-token rows) when by-channel.

3. **No per-token scale advance**: K_scales pointer is NOT advanced by `k0` in the k-loop prefetch (scale is the same for every iteration).

4. **V_scales not advanced**: `V_scales += ldvq * tile_m` is skipped when by-channel.

### Memory Layout Example (scale [1,10,1,128])

After `K_scales += KEY_COMP_OFF(b1, b0_kv, 0, 0)` = `head * 128`:
- `K_scales[0..127]` = 128 consecutive f16 scale values for this head
- Micro-kernel reads `K_scales[k]` for head_size position `k`, same value for all tokens
- Dequantization is performed on-the-fly inside the micro-kernel: `K_fp16[token][k] = K_int8[token][k] * K_scales[k]`

## How SDPA Organizes Layout

### `transpose_shape`: Computing Logical Shape

`transpose_shape(shape, order)` in `scaled_dot_product_attention.cpp` reorders a tensor's dimensions according to a permutation order — applying a logical transpose to the shape without moving data.

**Formula:** `shape_transposed[i] = shape[order[i]]`

**Example:** K physical shape `[1, 512, 10, 128]` with `input_k_transpose_order = {0, 2, 1, 3}`:

```
Input shape:  [1, 512, 10, 128]
               dim0  dim1  dim2  dim3

i=0: transposed[0] = shape[order[0]] = shape[0] = 1      (batch)
i=1: transposed[1] = shape[order[1]] = shape[2] = 10     (heads)
i=2: transposed[2] = shape[order[2]] = shape[1] = 512    (seq)
i=3: transposed[3] = shape[order[3]] = shape[3] = 128    (head_size)

Output shape: [1, 10, 512, 128]  →  [batch, heads, seq, head_size]
```

This computes what the kernel will "see" after the fused transpose remaps strides, used for output layout calculation.

### `convert_strides`: Mapping Physical Strides to Logical Dimensions

`convert_strides("KEY", "INPUT1", order)` in `sdpa_gen_micro.cpp` maps physical tensor strides to logical SDPA dimension strides using the fused transpose order as a permutation lookup table.

**Purpose:** The physical tensor layout has fixed strides (`BATCH_PITCH`, `FEATURE_PITCH`, `Y_PITCH`, `X_PITCH`), but the SDPA kernel needs strides in logical order (`S0`=batch, `S1`=heads, `S2`=seq, `S3`=head_size). When a transpose is fused, the physical-to-logical mapping changes.

**Formula:**

```
target[i] = source[order[i]]
```

Where `order[i]` says "logical dimension `i` corresponds to physical dimension `order[i]`."

### Example: K `[1, 512, 10, 128]` with `input_k_transpose_order = {0, 2, 1, 3}`

The physical layout is `[batch=1, seq=512, heads=10, head_size=128]` (seq and heads are swapped vs the standard SDPA convention `[batch, heads, seq, head_size]`).

**Physical strides** (from `make_layout_jit_constants("INPUT1", ...)`):
```
INPUT1_BATCH_PITCH   = 512 * 10 * 128 = 655360   (dim 0: batch)
INPUT1_FEATURE_PITCH = 10 * 128       = 1280      (dim 1: seq — physically in "feature" slot)
INPUT1_Y_PITCH       = 128                        (dim 2: heads — physically in "y" slot)
INPUT1_X_PITCH       = 1                          (dim 3: head_size)
```

**`convert_strides("KEY", "INPUT1", {0, 2, 1, 3})`** produces:
```
KEY_S0 = source[order[0]] = source[0] = INPUT1_BATCH_PITCH   = 655360  (logical batch → phys dim 0)
KEY_S1 = source[order[1]] = source[2] = INPUT1_Y_PITCH       = 128     (logical heads → phys dim 2)
KEY_S2 = source[order[2]] = source[1] = INPUT1_FEATURE_PITCH = 1280    (logical seq   → phys dim 1)
KEY_S3 = source[order[3]] = source[3] = INPUT1_X_PITCH       = 1      (logical hsize → phys dim 3)
```

**In sdpa_micro.cl**, the kernel uses:
```c
uint ldk = KEY_S2;  // stride between tokens = 1280 (not 128!)
```

This correctly navigates the physical memory: advancing one token means skipping 1280 elements (= 10 heads × 128 head_size), because tokens are stored along physical dim 1 (feature), not dim 2 (y).

### Key insight

No data is moved — the same physical memory is accessed. The transpose order just tells the kernel which physical stride to use for each logical SDPA dimension. This avoids an explicit transpose buffer while still letting the micro-kernel treat the data as `[batch, heads, seq, head_size]`.
