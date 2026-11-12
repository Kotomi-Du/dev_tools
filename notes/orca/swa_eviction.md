# Why https://github.com/intel-sandbox/winml.mtc.applications/pull/94/

Make usre the capcity is 1024, so it can use NPU AttentionDMA kernel; other than that,  no change for the attention behavior , minor difference of the eviction behavior at the first two chunks.

# Example of App-managed SWA eviction trace: W = 512, S = 512, prompt = 2060 tokens

- The prompt has 2060 tokens at positions 0–2059. Prefill runs as 4 full chunks plus a tail of 12 real tokens padded with 500 pad tokens. Decode #1 is at pos 2060.
- A query at pos p attends positions `[max(0, p − 511), p]`. This depends only on W, so the attended positions are **the same in all three versions**.
- Column meanings:
  - **Evict**: rows dropped during compaction.
  - **Kept**: rows kept after compaction.
  - **Rows read**: `total_seq_len_sliding`, the number of rows the model reads.
  - **Resident**: real tokens in the buffer after the call.
- Rows that are read but fall outside a query's window are masked by GQA.

## Summary

| | 1. Before PR | 2. PR as written | 3. Without lines 185–187 |
|---|---|---|---|
| Capacity | 1023 | 1024 | 1024 |
| Rows kept per prefill compaction | 511 | 511 | 512 |
| Rows read, chunk 1 | 1023 ✗ | 1024 ✓ | 1024 ✓ |
| Rows read, chunk 2 through tail | 1023 ✗ | 1023 ✗ | 1024 ✓ |
| Prefill compactions | 4 (chunk 1 through tail) | 3 (chunk 2 through tail) | 3 (chunk 2 through tail) |
| Decode compaction | every token | about every 513 tokens | every token |
| Attended positions | same | same | same (one extra row outside the window, masked) |

✓ / ✗ = whether the rows read are a multiple of 16, as the NPU's AttentionDMA kernel requires. This only matters if the compiler checks the row count passed at run time rather than the fixed 1024-row buffer size.

Possible fix that gets both benefits: change the lazy rule to keep W rows (`evict = max(evict, resident_length - window_size)`) and change the line 172 check to `capacity - S < W`. Prefill then behaves like version 3 (1024 rows read on every chunk), and decode compacts only about every 512 tokens.


## Attended positions (same in all three versions)

| Call | Absolute positions | Attended: first query → last query |
|---|---|---|
| chunk 0 | 0–511 | [0, 0] → [0, 511] |
| chunk 1 | 512–1023 | [1, 512] → [512, 1023] |
| chunk 2 | 1024–1535 | [513, 1024] → [1024, 1535] |
| chunk 3 | 1536–2047 | [1025, 1536] → [1536, 2047] |
| tail (12 real + 500 pad) | 2048–2059 | [1537, 2048] → [1548, 2059] |
| decode #1 | 2060 | [1549, 2060] |

## 1. Before PR (capacity = W + S − 1 = 1023, raw eviction only)

| Call | Resident before | Evict | Kept | Rows read | Resident after | Buffer after (row → pos) |
|---|---|---|---|---|---|---|
| chunk 0 | 0 | 0 | 0 | 512 | 512 | 0–511 → 0–511 |
| chunk 1 | 512 | **1** | 511 | 1023 ✗ | 1023 | 0–510 → 1–511, 511–1022 → 512–1023 |
| chunk 2 | 1023 | 512 | 511 | 1023 ✗ | 1023 | 0–510 → 513–1023, 511–1022 → 1024–1535 |
| chunk 3 | 1023 | 512 | 511 | 1023 ✗ | 1023 | 0–510 → 1025–1535, 511–1022 → 1536–2047 |
| tail | 1023 | 512 | 511 | 1023 ✗ | 523 | 0–510 → 1537–2047, 511–522 → 2048–2059, 523–1022 = pad |
| decode #1 | 523 | 0 | 523 | 524 | 524 | 523 → 2060 |
| decode at pos 2560 (step 500) and every later token | 1023 | **1** | 1022 | 1023 | 1023 | copies 1022 rows × 30 layers × K/V **per token** |

## 2. PR #94 as written (capacity 1024 + lazy eviction, keep W − 1 = 511)

| Call | Resident before | Evict (raw → lazy) | Kept | Rows read | Resident after | Buffer after (row → pos) |
|---|---|---|---|---|---|---|
| chunk 0 | 0 | 0 → 0 | 0 | 512 | 512 | 0–511 → 0–511 |
| chunk 1 | 512 | 0 → 0 | 512 | 1024 ✓ | 1024 | 0–1023 → 0–1023 |
| chunk 2 | 1024 | 512 → **513** | 511 | 1023 ✗ | 1023 | 0–510 → 513–1023, 511–1022 → 1024–1535 |
| chunk 3 | 1023 | 511 → **512** | 511 | 1023 ✗ | 1023 | 0–510 → 1025–1535, 511–1022 → 1536–2047 |
| tail | 1023 | 511 → **512** | 511 | 1023 ✗ | 523 | 0–510 → 1537–2047, 511–522 → 2048–2059, 523–1022 = pad |
| decode #1 | 523 | 0 → 0 | 523 | 524 | 524 | 523 → 2060 |
| decode at pos 2561 (step 501) | 1024 | 1 → **513** | 511 | 512 | 512 | next compaction **513 tokens later** |

## 3. Capacity 1024 without lines 185–187 (raw eviction only)

| Call | Resident before | Evict | Kept | Rows read | Resident after | Buffer after (row → pos) |
|---|---|---|---|---|---|---|
| chunk 0 | 0 | 0 | 0 | 512 | 512 | 0–511 → 0–511 |
| chunk 1 | 512 | 0 | 512 | 1024 ✓ | 1024 | 0–1023 → 0–1023 |
| chunk 2 | 1024 | 512 | **512** | 1024 ✓ | 1024 | 0–511 → 512–1023, 512–1023 → 1024–1535 |
| chunk 3 | 1024 | 512 | **512** | 1024 ✓ | 1024 | 0–511 → 1024–1535, 512–1023 → 1536–2047 |
| tail | 1024 | 512 | **512** | 1024 ✓ | 524 | 0–511 → 1536–2047, 512–523 → 2048–2059, 524–1023 = pad |
| decode #1 | 524 | 0 | 524 | 525 | 525 | 524 → 2060 |
| decode at pos 2560 (step 500) and every later token | 1024 | **1** | 1023 | 1024 | 1024 | copies 1023 rows × 30 layers × K/V **per token** |

