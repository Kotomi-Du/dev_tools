# GPU `statelessKV` flow for GQA

## Sequence-length flow

For GPU Group Query Attention (GQA), the decomposition creates `StatelessKV`
with the present-length convention:

```text
GQA seqlens_1d
    -> StatelessKV(..., true, window_size)
    -> GPU primitive is_seq_len_present_len = true
    -> present_seq_len = seqlens_1d
    -> past_seq_len = seqlens_1d - current_chunk_length
    -> sdpa_front_idx = max(past_seq_len - window_size + 1, 0)
    -> old KV = [sdpa_front_idx .. past_seq_len - 1]
    -> attened KV: old KV + new KV [past_seq_len .. present_seq_len - 1]
```

The GQA decomposition passes `true` for both the key and value caches. The
`seqlens_1d` input therefore represents the final/present sequence length,
not the length of the existing cache.

```cpp
StatelessKV(past_key, key, seqlens_1d, 2, true, window_size)
StatelessKV(past_value, value, seqlens_1d, value_seq_axis, true, window_size)
```

## Runtime state

The GPU runtime computes:

```cpp
if (is_seq_len_present_len) {
    present_seq_len = seq_len;
    past_seq_len = present_seq_len - current_chunk_length;
} else {
    past_seq_len = seq_len;
    present_seq_len = past_seq_len + current_chunk_length;
}
```

For GQA, the first branch is used.

## `concat_to_sdpa` in the rolling case

The rolling implementation uses three stages:

```text
1. concat_to_sdpa
   past cache + new KV -> output 1

2. sdpa_to_present (Here is when eviction happens)
   output 1 -> compact present cache output 0
```

The ordinary `new_to_present` stage is skipped while rolling, so the cache is
not overwritten before SDPA has received the KV it needs.

## Example

Given:

```text
cache capacity    = 1024
sliding window    = 512
present length    = 1069
current chunk     = 512
```

```
Step 1: process 0..511
Cache: [0..511]

Step 2: process 512..1023
Cache: [0..1023]

Step 3: replay 557..1068
Needed KV Cache [45...556] as full 512-token window for new chunk [557..1068]

But the cache can only keep the most recent resident KV rows after eviction:
        [513..1068]

KV [46..512] is lost due to eviction.
```

That's why the stage1 is required to save these old KV first

```text
past_seq_len = 1069 - 512 = 557
sdpa_front_idx = max(557 - 512 + 1, 0) = 46

old KV copied for SDPA = [46..556]
new KV                 = [557..1068]
output 1 for SDPA      = [46..1068]
```

# SWC GQA eviction policy
> Local window size can be fixed while chunk_size is changed, no impact for the conformance; because past KV can be attened in current iteration directly when chunk size is larger, no concern they are evicted earlier.

For **4096 tokens (`0..4095`)**, `lws = 512`, **right padded**, and `C = chunk_size + lws - 1` ; `G = C - LWZ + 1`. All ranges below are inclusive.
### Chunk size 512 → C = 1023

Eviction block size: `G = 512`.

| Step | Process tokens | Cache after step | KV evicted in this step |
|---|---|---|---|
| 1 | `0..511` | `0..511` | None |
| 2 | `512..1023` | `512..1023` | `0..511` |
| 3 | `1024..1535` | `1024..1535` | `512..1023` |
| 4 | `1536..2047` | `1536..2047` | `1024..1535` |
| 5 | `2048..2559` | `2048..2559` | `1536..2047` |
| 6 | `2560..3071` | `2560..3071` | `2048..2559` |
| 7 | `3072..3583` | `3072..3583` | `2560..3071` |
| 8 | `3584..4095` | `3584..4095` | `3072..3583` |

### Chunk size 1024 → C = 1535

Eviction block size: `G = 1024`.

| Step | Process tokens | Cache after step | KV evicted in this step |
|---|---|---|---|
| 1 | `0..1023` | `0..1023` | None |
| 2 | `1024..2047` | `1024..2047` | `0..1023` |
| 3 | `2048..3071` | `2048..3071` | `1024..2047` |
| 4 | `3072..4095` | `3072..4095` | `2048..3071` |

**Evicted does not mean unavailable during the whole step.** For example, query `2048` needs historical KV `1536..2047` in both configurations. Those keys must remain accessible until the queries needing them finish, even though they are removed from the final cache.


## Relevant source locations

- GQA creates `StatelessKV(..., true, window_size)` in
  `src/plugins/intel_gpu/src/plugin/transformations/group_query_attention_decomposition.cpp`.
- The GPU primitive copies the flag in
  `src/plugins/intel_gpu/src/plugin/ops/stateless_kv.cpp`.
- Runtime sequence lengths and `sdpa_front_idx` are calculated in
  `src/plugins/intel_gpu/src/graph/stateless_kv.cpp`.
- The rolling stages and `concat_to_sdpa` dispatch are implemented in
  `src/plugins/intel_gpu/src/graph/impls/ocl/stateless_kv.cpp`.
