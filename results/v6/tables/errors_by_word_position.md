Test set, W = 512, stride 128: top-1 accuracy by the target's position in its word ('space' = the target is a space, i.e. the end of a word; '1st letter' = the first letter of a word).

| Model | Target | Seeds | share of targets (%) | top-1 acc (%) | BPC | share of all errors (%) |
|---|---|---|---|---|---|---|
| ALiBi-3.2M (512, from scratch) | space | 3 | 17.10 ± 0.00 | 93.97 ± 0.01 | 0.3230 ± 0.0005 | 3.46 ± 0.00 |
| ALiBi-3.2M (512, from scratch) | 1st letter | 3 | 17.10 ± 0.00 | 28.57 ± 0.05 | 3.4759 ± 0.0017 | 41.04 ± 0.01 |
| ALiBi-3.2M (512, from scratch) | 2nd letter | 3 | 16.45 ± 0.00 | 58.97 ± 0.06 | 1.7392 ± 0.0024 | 22.67 ± 0.03 |
| ALiBi-3.2M (512, from scratch) | 3rd letter | 3 | 14.10 ± 0.00 | 69.18 ± 0.03 | 1.4148 ± 0.0017 | 14.60 ± 0.01 |
| ALiBi-3.2M (512, from scratch) | 4th+ letter | 3 | 35.24 ± 0.00 | 84.61 ± 0.02 | 0.6926 ± 0.0004 | 18.22 ± 0.03 |
| ALiBi-3.2M (512, from scratch) | all | 3 | 100 | 70.23 ± 0.02 | 1.3795 ± 0.0008 | 100 |
| ALiBi-14.2M (512, from scratch) | space | 3 | 17.10 ± 0.00 | 94.68 ± 0.02 | 0.2714 ± 0.0006 | 3.39 ± 0.02 |
| ALiBi-14.2M (512, from scratch) | 1st letter | 3 | 17.10 ± 0.00 | 31.69 ± 0.04 | 3.3241 ± 0.0004 | 43.60 ± 0.02 |
| ALiBi-14.2M (512, from scratch) | 2nd letter | 3 | 16.45 ± 0.00 | 62.80 ± 0.06 | 1.5805 ± 0.0011 | 22.83 ± 0.02 |
| ALiBi-14.2M (512, from scratch) | 3rd letter | 3 | 14.10 ± 0.00 | 73.23 ± 0.02 | 1.2272 ± 0.0010 | 14.09 ± 0.00 |
| ALiBi-14.2M (512, from scratch) | 4th+ letter | 3 | 35.24 ± 0.00 | 87.76 ± 0.03 | 0.5560 ± 0.0016 | 16.10 ± 0.02 |
| ALiBi-14.2M (512, from scratch) | all | 3 | 100 | 73.20 ± 0.03 | 1.2440 ± 0.0008 | 100 |
| Transformer (ALiBi) + 512-window fine-tuning | space | 3 | 17.10 ± 0.00 | 93.73 ± 0.04 | 0.3378 ± 0.0020 | 3.53 ± 0.02 |
| Transformer (ALiBi) + 512-window fine-tuning | 1st letter | 3 | 17.10 ± 0.00 | 27.62 ± 0.08 | 3.5151 ± 0.0045 | 40.73 ± 0.06 |
| Transformer (ALiBi) + 512-window fine-tuning | 2nd letter | 3 | 16.45 ± 0.00 | 58.12 ± 0.12 | 1.7731 ± 0.0061 | 22.66 ± 0.01 |
| Transformer (ALiBi) + 512-window fine-tuning | 3rd letter | 3 | 14.10 ± 0.00 | 68.43 ± 0.14 | 1.4505 ± 0.0055 | 14.65 ± 0.04 |
| Transformer (ALiBi) + 512-window fine-tuning | 4th+ letter | 3 | 35.24 ± 0.00 | 84.09 ± 0.04 | 0.7160 ± 0.0025 | 18.44 ± 0.02 |
| Transformer (ALiBi) + 512-window fine-tuning | all | 3 | 100 | 69.60 ± 0.07 | 1.4075 ± 0.0034 | 100 |
| Transformer (RoPE) + 512-window fine-tuning | space | 3 | 17.10 ± 0.00 | 93.65 ± 0.06 | 0.3429 ± 0.0039 | 3.56 ± 0.01 |
| Transformer (RoPE) + 512-window fine-tuning | 1st letter | 3 | 17.10 ± 0.00 | 27.42 ± 0.12 | 3.5225 ± 0.0042 | 40.61 ± 0.22 |
| Transformer (RoPE) + 512-window fine-tuning | 2nd letter | 3 | 16.45 ± 0.00 | 57.95 ± 0.18 | 1.7784 ± 0.0068 | 22.62 ± 0.06 |
| Transformer (RoPE) + 512-window fine-tuning | 3rd letter | 3 | 14.10 ± 0.00 | 68.32 ± 0.12 | 1.4536 ± 0.0066 | 14.62 ± 0.05 |
| Transformer (RoPE) + 512-window fine-tuning | 4th+ letter | 3 | 35.24 ± 0.00 | 83.87 ± 0.38 | 0.7225 ± 0.0187 | 18.59 ± 0.31 |
| Transformer (RoPE) + 512-window fine-tuning | all | 3 | 100 | 69.43 ± 0.21 | 1.4133 ± 0.0099 | 100 |
