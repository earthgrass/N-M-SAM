# APBR Parent Magnitude vs Geometry — Preflight Audit

Status: **PASS**

The three fixed APBR-SAM Epoch-200 parents, matching geometry states, Geometry Raw/Recovery checkpoints, masks, and final evaluations were verified read-only. Existing Geometry recovery can be reused for a paired ablation because loader order and augmentation worker seeds are explicit functions of `(seed, epoch)`; all 20 recorded order hashes match.

| Seed | APBR parent SHA256 | Geometry mask SHA256 | Raw | Recovery-20 |
|---:|---|---|---:|---:|
| 0 | `4332079becdb58fb50f0e68b8bd0dcc251db0184a6050facb524817520a51d82` | `79d9c410c67e931101cf46112e732c68133a9935e4f99a41503a0de8ee7b40f4` | 59.22% | 78.72% |
| 1 | `6571e30ed783508f28bc378aa3b22a739bf1ddbbb790db3d98c08858e126e10b` | `54eedcd0449582a0afb1d240e41bb8223e36536e62e90539bab8cfce421fd6ac` | 51.87% | 79.15% |
| 2 | `6b55caf932d846e16a13f7c2510e216db790bcd6738941288491c2e1d4b24a22` | `53129860937225952922d6f79169e4e138f2667744bb12314085303cb741620b` | 56.14% | 78.71% |

Each Geometry mask has 19 layers, 11,157,504 eligible positions, and 5,578,752 deleted positions. Survivors equal their dense-parent values before recovery; deleted values are exactly zero.

The new Magnitude branch will use the identical OIHW 2:4 implementation, Recovery-20 hyperparameters, and final-only Test evaluator in an isolated ablation directory.
