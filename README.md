# FC‑LoRA  ⚡ Attention‑Only Fisher‑Covariance LoRA for DeiT

> **Train <0.3 % of DeiT‑Base, keep ≥98 % of its accuracy.**

---

## ✨ Highlights

|                                |                                                                 |
| ------------------------------ | --------------------------------------------------------------- |
| **Attention‑Only**             | Adapters added **only** to Q/K/V projections – no MLP or norm mods |
| **Fisher × Covariance Scoring**| Layer importance = 0.6·Fisher + 0.4·Covariance                    |
| **Rank Budgeting**             | Water‑fill 100 total ranks, ≤16 per layer                         |
| **Tiny Foot‑print**            | 198 k / 86 M trainable → **0.23 %**                               |
| **Fast**                       | 15 epochs ⇒ 48 min on one RTX 3090                                |
| **Well‑Calibrated**            | Scaled ECE ≈ 0.015 (CIFAR‑100)                                    |

---

## How It Works

- **Attention-Only LoRA:** This method injects LoRA adapters *exclusively* into the query, key, and value (Q/K/V) projection layers of the DeiT transformer blocks. No modifications are made to MLPs or normalization layers.
- **Layer Importance Scoring:** Each attention layer’s “importance” is measured by a weighted combination of the diagonal Fisher information (0.6 weight) and the activation covariance trace (0.4 weight), both estimated from a short warmup on the training set.
- **Rank Allocation:** A total rank budget of 100 is distributed across the Q/K/V layers (up to 16 per layer), with higher-importance layers receiving more ranks. All other parameters remain frozen.
- **Tiny Trainable Footprint:** Only ~0.23% of all parameters are updated (198k out of 86M for DeiT‑Base), yet accuracy loss is minimal.
- **Training Setup:** Models are trained for 15 epochs (CIFAR-100) or 10 epochs (CIFAR-10) using CutMix, cosine LR schedule, and label smoothing.

---

## Results

### CIFAR‑100 – 15 epochs

| Metric               | FC‑LoRA                | Full Finetune        |
| -------------------- | ---------------------- | -------------------- |
| **Trainable params** | **198 144 (0.23 %)**   | 86 075 236 (100 %)   |
| **Test Accuracy**    | **89.19 %**            | 90.18 %              |
| **Test Loss**        | 1.1284                 | 1.1345               |
| **ECE / Scaled ECE** | 0.0920 / **0.0146**    | 0.0560 / 0.0266      |
| **Class Acc µ±σ**    | 0.89 ± 0.08            | 0.90 ± 0.07          |
| **Wall‑clock**       | **2 862 s (~48 min)**  | 6 821 s (~1 h 54 m)  |

### CIFAR‑10 – 10 epochs

| Metric               | FC‑LoRA                | Full Finetune      |
| -------------------- | ---------------------- | ------------------ |
| **Trainable params** | **125 952 (0.15 %)**   | 85 933 834 (100 %) |
| **Test Accuracy**    | **98.27 %**            | 98.51 %            |
| **ECE / Scaled ECE** | 0.0886 / **0.0045**    | 0.0840 / 0.0051    |
| **Wall‑clock**       | **1 903 s (~32 min)**  | 2 279 s (~38 min)  |
