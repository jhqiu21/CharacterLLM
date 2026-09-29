# Numbers behind the paper's claims
Mean ± sample std over seeds; 'diff' = seed-matched difference (same seed index), mean ± std over the 3 pairs.

## (a) TR2o vs parameter-matched LR-2: BPC (full_context)
| model | BPC |
|---|---|
| TR2o | 1.6132 ± 0.0174 |
| LR-2 | 1.7009 ± 0.0011 |

- TR2o − LR-2: -0.0877 ± 0.0164 (per seed: -0.0945, -0.0689, -0.0995)

## (a') same, legacy32 BPC
| model | BPC |
|---|---|
| TR2o | 1.8427 ± 0.0031 |
| LR-2 | 1.8855 ± 0.0007 |

- TR2o − LR-2: -0.0428 ± 0.0034 (per seed: -0.0425, -0.0463, -0.0395)

## (b) Calibration: ECE (full_context)
| model | ECE |
|---|---|
| TR2o | 0.0098 ± 0.0007 |
| TR8o | 0.0318 ± 0.0002 |
| LR-1 | 0.0254 ± 0.0003 |
| LR-2 | 0.0262 ± 0.0000 |

- TR2o − LR-1: -0.0156 ± 0.0009 (per seed: -0.0162, -0.0161, -0.0146)

- TR2o − LR-2: -0.0165 ± 0.0006 (per seed: -0.0167, -0.0169, -0.0157)

- TR8o − LR-1: 0.0064 ± 0.0003 (per seed: +0.0062, +0.0066, +0.0065)

- TR8o − LR-2: 0.0056 ± 0.0002 (per seed: +0.0057, +0.0058, +0.0054)

## (c) Self-BLEU-4 (lower = more diverse)
| model | Self-BLEU-4 |
|---|---|
| TR2o | 0.1577 ± 0.0200 |
| TR8o | 0.1165 ± 0.0191 |
| LR-1 | 0.1482 ± 0.0098 |
| LR-2 | 0.1697 ± 0.0151 |

- TR2o − TR8o: 0.0412 ± 0.0112 (per seed: +0.0319, +0.0537, +0.0380)

- TR2o − LR-1: 0.0095 ± 0.0129 (per seed: -0.0054, +0.0169, +0.0170)

- TR2o − LR-2: -0.0119 ± 0.0132 (per seed: -0.0258, +0.0005, -0.0105)

## (d) RoPE (TR2o) vs relative (TR8o): BPC (full_context)
| model | BPC |
|---|---|
| TR2o | 1.6132 ± 0.0174 |
| TR8o | 1.7063 ± 0.0054 |

- TR2o − TR8o: -0.0931 ± 0.0201 (per seed: -0.1071, -0.0700, -0.1021)

## (d') same, legacy32 BPC
| model | BPC |
|---|---|
| TR2o | 1.8427 ± 0.0031 |
| TR8o | 1.8886 ± 0.0016 |

- TR2o − TR8o: -0.0459 ± 0.0016 (per seed: -0.0455, -0.0477, -0.0445)

## (e) Ablation, legacy32 BPC
| model | BPC |
|---|---|
| TR2o-legacy | 1.8457 ± 0.0021 |
| TR2o-rngfix | 1.8460 ± 0.0029 |
| TR2o | 1.8427 ± 0.0031 |

- TR2o-rngfix − TR2o-legacy: 0.0003 ± 0.0012 (per seed: -0.0005, -0.0002, +0.0017)

- TR2o − TR2o-rngfix: -0.0034 ± 0.0003 (per seed: -0.0037, -0.0033, -0.0031)

- TR2o − TR2o-legacy: -0.0030 ± 0.0015 (per seed: -0.0042, -0.0035, -0.0014)

## (e) Ablation, chunked128 BPC
| model | BPC |
|---|---|
| TR2o-legacy | 1.6924 ± 0.0072 |
| TR2o-rngfix | 1.6891 ± 0.0058 |
| TR2o | 1.6835 ± 0.0068 |

- TR2o-rngfix − TR2o-legacy: -0.0033 ± 0.0080 (per seed: -0.0125, +0.0010, +0.0016)

- TR2o − TR2o-rngfix: -0.0056 ± 0.0010 (per seed: -0.0061, -0.0044, -0.0062)

- TR2o − TR2o-legacy: -0.0089 ± 0.0084 (per seed: -0.0186, -0.0034, -0.0046)

## (e) Ablation, full_context BPC
| model | BPC |
|---|---|
| TR2o-legacy | 1.6278 ± 0.0160 |
| TR2o-rngfix | 1.6208 ± 0.0155 |
| TR2o | 1.6132 ± 0.0174 |

- TR2o-rngfix − TR2o-legacy: -0.0070 ± 0.0151 (per seed: -0.0244, +0.0026, +0.0007)

- TR2o − TR2o-rngfix: -0.0076 ± 0.0020 (per seed: -0.0084, -0.0053, -0.0090)

- TR2o − TR2o-legacy: -0.0146 ± 0.0160 (per seed: -0.0327, -0.0027, -0.0083)

## (e') Evaluation context: TR2o BPC by mode
| model | BPC |
|---|---|
| TR2o | 1.6132 ± 0.0174 |
- TR2o legacy32: 1.8427 ± 0.0031
- TR2o chunked128: 1.6835 ± 0.0068
- TR2o full_context: 1.6132 ± 0.0174

## (f) Per-step speed: LSTM s/step ÷ Transformer s/step (solo runs only; see throughput.md)
| model | solo runs | s/step (solo mean) |
|---|---|---|
| TR2o | 2 | 0.0207 |
| TR8o | 3 | 0.0185 |
| LR-1 | 2 | 0.0517 |
| LR-2 | 2 | 0.0654 |
- LR-1 / TR2o: 2.50x
- LR-2 / TR2o: 3.17x
- LR-1 / TR8o: 2.80x
- LR-2 / TR8o: 3.54x

## ECE provenance check: validation ECE at iteration 0 (untrained model) vs at the selected checkpoint
The original Logger.print_metrics printed `self.ece[0]` (the iteration-0 value) at every evaluation.

| model | val ECE at iteration 0 (per seed) | val ECE at best checkpoint | test ECE (full_context) | paper Table V ECE |
|---|---|---|---|---|
| TR2o | 0.1812, 0.0956, 0.1366 | 0.0103 ± 0.0010 | 0.0098 ± 0.0007 | 0.083 |
| TR8o | 0.1635, 0.1032, 0.1392 | 0.0295 ± 0.0007 | 0.0318 ± 0.0002 | 0.1322 |
| LR-1 | 0.0018, 0.0103, 0.0042 | 0.0218 ± 0.0008 | 0.0254 ± 0.0003 | 0.0227 |
| LR-2 | 0.0548, 0.0131, 0.0118 | 0.0222 ± 0.0004 | 0.0262 ± 0.0000 | 0.026 |
| TR2o-legacy | 0.1812, 0.0956, 0.1366 | 0.0105 ± 0.0005 | 0.0095 ± 0.0005 | 0.083 |
