# PR #39814 – `ScaleDownGatedMLPInput` (GPU, FP16 gated MLP overflow)

PR: https://github.com/openvinotoolkit/openvino/pull/39814
## 0. review learning
1. check if new transformation would impact any transfromation pattern match afterwards
2. consider a node with different cases of inputs

## 1. What the graph looks like after the transformation

The pass adds and removes **no nodes**. It only replaces some constant inputs of the existing
`FullyConnectedCompressed` (FCC) nodes with folded copies named `*_gated_mlp_scaled`. Default factor = 8.

### Separate gate / up FCs
```
Before:
          ┌─ FCC(gate) [W, bias, scale_g, zp] ── Swish ─┐
 input ───┤                                             Multiply (f16) ── FCC(down) [W, bias_d, scale_d, zp] ── out
          └─ FCC(up)   [W, bias_u, scale_u, zp] ────────┘

After:
          ┌─ FCC(gate) [W, bias, scale_g, zp] ─────────── Swish ─┐
 input ───┤                                                      Multiply (f16, 8x smaller) ── FCC(down) [W, bias_d, scale_d*8, zp] ── out (unchanged)
          └─ FCC(up)   [W, bias_u/8, scale_u/8, zp] ─────────────┘
```

### Merged gate|up (after `FullyConnectedHorizontalFusion`)
```
 input ── FCC(gate|up) ── VariadicSplit(axis=-1) ─┬─ gate ── Swish ─┐
                                                  └─ up ────────────┴─ Multiply ── FCC(down) [scale_d*8]
```
- Only the up channels `[begin, end)` of the merged scale and bias are multiplied by 1/8, using a per-channel factor `[1,…,1, 1/8,…,1/8]`.
- The topology doesn't change, so `GLUFusion` still produces SwiGLU.

### Ops allowed on the up branch
- `Reshape`, `Squeeze`, `Unsqueeze`, float-to-float `Convert`: passed through unchanged
- `Add` with a constant: the constant is scaled by 1/8

### What changes
| Input | Change |
|---|---|
| up FC decompression scale (input 3) | × 1/8 (per channel if merged) |
| up FC bias (input 2, if not `Placeholder`) | × 1/8 |
| constant of each `Add` on the up branch | × 1/8 |
| down FC decompression scale (input 3) | × 8 |
| down FC bias | unchanged |

### When the pass skips a match
- The product isn't f16, or it has more than one consumer
- Neither operand of the `Multiply` is a `Swish`
- Something on the up branch has more than one consumer, or an `Add` doesn't have exactly one constant input
- The up projection isn't an FCC, or the split isn't on the last axis
- A scaled FP16 constant would become inf or subnormal (all folds are computed before anything is replaced)

### Where it runs
Registered in `transformations_pipeline.cpp` after the horizontal FC and LoRA fusions.
Only when `infer_precision == f16` and `ACTIVATIONS_SCALE_FACTOR` isn't set.

---

## 2. Why ÷8 / ×8 helps

### Where the overflow is
```
h   = Swish(gate) * up    ← stored as f16; one element > 65504 → inf
out = down(h)             ← inf in the dot product → inf - inf → NaN, for the whole row
```
The final output values are in range. Only the intermediate `h` overflows (`up` reaches about 311).

### Moving the constant across the FCs (they're linear)
```
down(Swish(gate) * up) = (8·down)(Swish(gate) * (up/8))
```
- **up/8**: makes `h` 8x smaller, so a value of ~70000 (inf) becomes ~8750 → 8x headroom.
- **down·8**: restores the size inside the down FC. The dot product accumulates at higher precision and the true outputs are in range.
- The result is mathematically identical.

### Design points
- **Power of two**: scaling only changes the exponent, so it's exact (see section 3).
- **Only up is scaled, not gate**: Swish isn't linear (`Swish(x/8) ≠ Swish(x)/8`).
- **The down FC's bias is unchanged**: it's added after the matmul, whose output size is restored.
  The up FC's bias and the up-branch `Add` constants are part of `up`, so they're divided by 8.
- **Why 8**: the smallest power of two that comfortably covers the observed overflow, at the cost of
  3 exponent steps at the low end. The subnormal check keeps the constants exact.
- **No runtime cost**: the factors are folded into constants at compile time, so there are no extra kernels.
  Earlier alternatives (explicit `Multiply` nodes, or `ACTIVATIONS_SCALE_FACTOR` on every MatMul) cost TTFT.

---

## 3. FP16 representation: exponent and mantissa

```
 bit: 15 | 14 13 12 11 10 | 9 8 7 6 5 4 3 2 1 0
      S  |   exponent (E) |    mantissa (M)

 value = (−1)^S × 1.M × 2^(E − 15)
```
- **S**: sign
- **E** (5 bits, bias 15): sets how large the number is
- **M** (10 bits, implicit leading 1): sets the precision

### Examples
| Value | S | E | M | Calculation |
|---|---|---|---|---|
| 1.0 | 0 | 01111 (15) | 0000000000 | 1.0 × 2⁰ |
| 3.0 | 0 | 10000 (16) | 1000000000 | 1.5 × 2¹ |
| 0.125 | 0 | 01100 (12) | 0000000000 | 1.0 × 2⁻³ |
| 65504 (max) | 0 | 11110 (30) | 1111111111 | 1.999 × 2¹⁵ |

### Special cases
| E | M | Meaning |
|---|---|---|
| 31 | 0 | ±inf (overflow) |
| 31 | ≠ 0 | NaN |
| 0 | 0 | ±0 |
| 0 | ≠ 0 | subnormal: `0.M × 2⁻¹⁴`, less precision |

Smallest normal value = `2⁻¹⁴ = 6.103515625e-05`, the `min_normal_f16` constant in the pass.

### Why ×8 / ÷8 is exact
8 = 2³, so scaling only adds or subtracts 3 from E. S and M don't change, so there's no rounding:
```
 3.0     = 0 | 10000 | 1000000000   (1.5 × 2¹)
 3.0 / 8 = 0 | 01101 | 1000000000   (1.5 × 2⁻²) = 0.375
 3.0 * 8 = 0 | 10011 | 1000000000   (1.5 × 2⁴)  = 24
```
This only holds while 1 ≤ E ≤ 30:
- E > 30 → **inf** (the overflow this PR fixes)
- E < 1 → **subnormal**: mantissa bits are shifted out and precision is lost

So `fold_scaled` rejects any scaled FP16 constant that is inf or non-zero but below `6.1e-05`.
The test `ScaleDownGatedMLPInput_SubnormalScale_NotApplied` checks this with `1e-4 / 8 ≈ 1.25e-05`.