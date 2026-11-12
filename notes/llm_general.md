# transformer
```

## level 0
Input tokens
     │
     ▼
┌───────────────┐
│ Self-Attention│  ← tokens communicate with each other
└───────────────┘
     │
     ▼
┌───────────────┐
│      FFN      │  ← transforms each token individually
└───────────────┘
     │
     ▼
   Output


## level 1
Input hidden states X
        │
        ▼
   RMSNorm / LayerNorm
        │
        ├── Q projection: X × Wq → Q
        ├── K projection: X × Wk → K
        └── V projection: X × Wv → V
                │
                ▼
     Attention(Q, K, V)
                │
                ▼
       Output projection Wo
                │
                ▼
        + Residual connection
                │
                ▼
              FFN ->├─ Gate projection
                    ├─ Up projection
                    └─ Down projection
```
# MOE 
Assume there are 8 experts.

The router usually contains a small learned linear projection: 
$$
\text{logits} = xW_{\text{router}}
$$

where:

$$
W_{\text{router}} \in \mathbb{R}^{4096 \times 8}
$$
Therefore:

$$
[1 \times 4096] \times [4096 \times 8]
\rightarrow [1 \times 8]
$$

The result contains one score for each expert:

$$
\text{scores} =
[s_0, s_1, s_2, \ldots, s_7]
$$

For **Top-2 routing**, the two experts with the highest scores are selected.

```
   Attention
   ↓
 Router
 ╱ │ ╲
E0 E1 E2 ...  ← multiple FFNs
```