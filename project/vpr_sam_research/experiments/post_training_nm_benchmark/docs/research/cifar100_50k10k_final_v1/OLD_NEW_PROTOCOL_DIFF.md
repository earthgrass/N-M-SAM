# Old/new formal protocol differences

| Field | Aborted 45k/5k protocol | Frozen 50k/10k protocol |
|---|---|---|
| Official Train usage | 45,000 train + 5,000 validation | all 50,000 train |
| Validation size | 5,000 | 0 |
| Samples per epoch | 45,000 | 50,000 |
| Batches per epoch | 352 | 391 |
| `drop_last` | false | false |
| Normalization source | 45,000 subset | official full 50,000 Train |
| Normalization mean | 0.507485 / 0.486780 / 0.441163 | 0.507075 / 0.486549 / 0.440918 |
| Model selection | maximum validation accuracy, earliest tie | fixed epoch 200 Final |
| Dense primary file | `best_val.pt` | `final_epoch_200.pt` |
| Test during training | not loaded | not loaded |
| Recovery selection | maximum validation accuracy | fixed epoch 20 Final |
| Recovery primary file | `best_val.pt` | `final_epoch_020.pt` |
| Resume compatibility | old protocol only | old protocol explicitly rejected |
| Output root | research 45k/5k RUNS directory | `artifacts/cifar100_50k10k_final_v1/` |

Optimizer, learning rate, momentum, weight decay, SAM radius, scheduler,
augmentation, batch size, precision, APBR geometry, N:M layout and method
adapters are unchanged by this reset.
