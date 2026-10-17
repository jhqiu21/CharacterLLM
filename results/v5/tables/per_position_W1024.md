Per-position test BPC at W=1024: mean NLL over 2,048 non-overlapping test chunks (seed 0), averaged over the positions of each bucket and over 3 seeds.

| Model | 0–63 | 64–127 | 128–191 | 192–255 | 448–511 | 512–575 | 960–1023 |
|---|---|---|---|---|---|---|---|
| RoPE | 1.617 | 1.453 | 3.506 | 5.183 | 4.884 | 4.871 | 4.440 |
| RoPE + PI | 3.964 | 4.041 | 4.049 | 4.050 | 4.043 | 4.057 | 4.053 |
| RoPE + local attention 128 | 1.617 | 1.453 | 1.445 | 1.448 | 1.443 | 1.463 | 1.447 |
| RoPE + SelfExtend floor, near 64 | 1.617 | 1.452 | 1.436 | 1.432 | 1.423 | 1.447 | 1.456 |
| RoPE + SelfExtend linear, near 64 | 1.617 | 1.452 | 1.436 | 1.432 | 1.423 | 1.447 | 1.462 |
| RoPE + 512-window fine-tuning | 1.638 | 1.470 | 1.452 | 1.443 | 1.423 | 1.449 | 1.531 |
| RoPE + PI + 512-window fine-tuning | 1.684 | 1.513 | 1.491 | 1.480 | 1.447 | 3.709 | 5.140 |
| ALiBi + 512-window fine-tuning | 1.636 | 1.463 | 1.441 | 1.429 | 1.400 | 1.420 | 1.396 |
| ALiBi | 1.633 | 1.462 | 1.443 | 1.432 | 1.409 | 1.429 | 1.407 |
