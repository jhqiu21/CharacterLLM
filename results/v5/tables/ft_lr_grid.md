Fine-tuning learning-rate grid, seed 42, 1,000 iterations (warmup 50, cosine to 0): best fixed-validation BPC (1024 validation windows of the condition's training length: 512, or 128 for the 128-window condition). * = chosen (lowest). 'before fine-tuning': the same validation set at iteration 0 (the first step has lr 0, so this is the source model). (it. 0) = the best checkpoint is iteration 0, i.e. fine-tuning never improved on the source model; 3e-3 was added after the first four (the PI condition's best was 1e-3, the upper edge).

| Condition | before fine-tuning | lr 3e-5 | lr 1e-4 | lr 3e-4 | lr 1e-3 | lr 3e-3 |
|---|---|---|---|---|---|---|
| RoPE + PI + 512-window fine-tuning | 4.0353 | 2.2687 | 1.8156 | 1.5808 | 1.5402 * | 1.5998 |
| RoPE + 512-window fine-tuning | 3.9224 | 1.9630 | 1.6694 | 1.5069 * | 1.5193 | 1.5716 |
| RoPE + 128-window fine-tuning | 1.5757 | 1.5757 (it. 0) | 1.5757 (it. 0) | 1.5757 (it. 0) | 1.5757 (it. 0) | 1.5757 (it. 0) |
| ALiBi + 512-window fine-tuning | 1.4865 | 1.4819 * | 1.4822 | 1.4860 | 1.4865 (it. 0) | 1.4865 (it. 0) |
