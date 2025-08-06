# Single-score DTI benchmark results

- Illiadis: https://doi.org/10.1186/s12859-024-05684-y
- TxGemma: https://arxiv.org/abs/2504.06196
- SSM-DTA: https://doi.org/10.1093/bib/bbad386 (very similar goals as our thesis; they note on pretraining strategies)
- MMELON (IBM Biomed): https://arxiv.org/abs/2410.19704

Baseline & multi-modal model results were more extensively tuned and the average test-set results over the 5 best configurations were reported.The other models, due to pre-training, were not tuned as extensively and the results report the best test-set results of the single best configuration (based on the validation set).

## DAVIS (pKd)

### Random Split

| Model        | MSE ↓  | RMSE ↓ | R2 ↑  | CI ↑  | Pearson ↑ | Params |
|--------------|--------|--------|--------|--------|---------|--------|
| Baseline     | 0.2352 | 0.4849 | 0.6924 | 0.8896 | 0.8324  | 13.9M  |
| Multi-modal  | 0.2401 | 0.4898 | 0.6858 | 0.8806 | 0.8286  | 7.7M   | **REDO**
| Multi-output | 0.2585 | 0.5085 | 0.6941 | 0.8776 | 0.8414  | 44.3M  |
| Multi-hybrid | 0.2153 | 0.4640 | 0.7384 | 0.9001 | 0.8702  | 65.9M  |
| Full         | | | | | |  |
|--------------|--------|--------|--------|--------|---------|--------|
| Iliadis      | 0.2656 |        | 0.6733 | 0.8702 |         | 2.1M    | (MLP-MLP)
| SSM-DTA      | 0.219  |        |        | 0.890  |         |         |
| MMELON       |        |        |        |        |         | 84M     |


### Cold Split

| Model        | MSE ↓  | RMSE ↓ | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|--------------|--------|--------|--------|--------|---------|--------|
| Baseline     | 0.8609 | 0.9274 | 0.3096 | 0.7664 | 0.5677  | 1.7M   |
| Multi-modal  | 0.7538 | 0.8681 | 0.3955 | 0.8003 | 0.6437  | 2.9M   |
| Multi-output | 0.9612 | 0.9804 | 0.3861 | 0.7561 | 0.6510  | 45.1M  |
| Multi-hybrid | 0.8860 | 0.9413 | 0.2895 | 0.7346 | 0.5858  | 65.9M  |
| Full         |        |        |        |        |         |        |
|--------------|--------|--------|--------|--------|---------|--------|
| Iliadis      | 0.6189 |        |-0.0426 | 0.7304 |         | 6.3M   | (MLP-MLP)
| TxGemma      | 0.555  |        |        |        |         | 27B    | (27B-Predict)
| SSM-DTA      | 0.8019 |        | 0.2803 |        |         |        |

## KIBA (KIBA score)

### Random Split

| Model        | MSE ↓  | RMSE ↓ | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|--------------|--------|--------|--------|--------|-----------|--------|
| Baseline     | 0.1573 | 0.3966 | 0.7648 | 0.8595 | 0.8749    | 16.9M  |
| Multi-modal  | 0.1955 | 0.4422 | 0.7074 | 0.8479 | 0.8412    | 10.1M  | **REDO**
| Multi-output | 0.2358 | 0.4856 | 0.6475 | 0.8389 | 0.8213    | 44.3M  |
| Multi-hybrid | 0.2104 | 0.4587 | 0.6853 | 0.8618 | 0.8536    | 65.9M  |
| Full | | | | | |  |
|--------------|--------|--------|--------|--------|---------|--------|
| Iliadis      | 0.1994 |        | 0.7187 | 0.8379 |         | 2.1M   | (MLP-MLP)
| SSM-DTA      | 0.154  |        |        | 0.895  |         |        |

### Cold Split

| Model        | MSE ↓  | RMSE ↓ | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|--------------|--------|--------|--------|--------|-----------|--------|
| Baseline     | 0.3702 | 0.6084 | 0.3899 | 0.7335 | 0.6481    | 15.4M  |
| Multi-modal  | 0.4120 | 0.6418 | 0.3210 | 0.7243 | 0.5824    | 31.1M  |
| Multi-output | 0.4535 | 0.6734 | 0.2525 | 0.7171 | 0.5808    | 45.1M  |
| Multi-hybrid | 0.4701 | 0.6856 | 0.2252 | 0.7374 | 0.5883    | 65.9M  |
| Full | | | | | |  |
|--------------|--------|--------|--------|--------|---------|--------|
| Iliadis      | 0.4167 |        | 0.4498 | 0.7510 |         | 9.9M   | (MLP-MLP)
| TxGemma      | 0.588  |        |        |        |         | 9B     | (9B-Predict)


# Multi-score DTI results (real-valued and binary)

### Random Split

| Model | MSE pKd ↓ | MSE pKi ↓ | MSE KIBA ↓ | Accuracy ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ | Params |
|-------|-----|------|----|----|---------|--------|---------|--------|
| Multi-output | 0.4212 | 0.6633 | 0.2358 | 0.8742 | 0.7679 | 0.9253 | 0.8451 | 44.3M |
| Multi-hybrid | 0.3720 | 0.7014 | 0.2104 | 0.8627 | 0.7556 | 0.9225 | 0.8320 | 65.9M |
| Full | | | | | | | | |

### Cold Split

| Model | MSE pKd ↓ | MSE pKi ↓ | MSE KIBA ↓ | Accuracy ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ | Params |
|-------|-----|------|----|----|---------|--------|---------|--------|
| Multi-output | 0.9612 | 0.8288 | 0.4535 | 0.8458 | 0.6929 | 0.8816 | 0.7713 | 45.1M |
| Multi-hybrid | 1.2683 | 0.8862 | 0.4701 | 0.8300 | 0.6785 | 0.8743 | 0.7446 | 65.9M |
| Full | | | | | | | | |

# Comparisons

## Simple residual encoder vs. transformer encoder

For baseline model:

| Dataset | Split | MLP        | MSE ↓  | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|---------|-------|------------|--------|--------|--------|-----------|--------|
| Davis   | Random| Residual   | 0.2324 | 0.6960 | 0.8920 | 0.8349    | 15M    |
|         |       | Transformer| 0.2346 | 0.6931 | 0.8903 | 0.8328    | 14M    |
|         | Cold  | Residual   | 0.8914 | 0.2851 | 0.7572 | 0.5460    | 1.6M   |
|         |       | Transformer| 0.8390 | 0.3271 | 0.7629 | 0.5740    | 1.8M   |
| KIBA    | Random| Residual   | 0.1606 | 0.7598 | 0.8580 | 0.8721    | 16.5M  |
|         |       | Transformer| 0.1564 | 0.7661 | 0.8594 | 0.8756    | 17.2M  |
|         | Cold  | Residual   | 0.3661 | 0.3967 | 0.7358 | 0.6519    | 16.7M  |
|         |       | Transformer| 0.3715 | 0.3877 | 0.7324 | 0.6461    | 15.6M  |

For multi-modal model:

| Dataset | Split | MLP        | MSE ↓  | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|---------|-------|------------|--------|--------|--------|-----------|--------|
| Davis   | Random| Residual   | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         |       | Transformer| x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | Residual   | 0.7538 | 0.3955 | 0.8003 | 0.6437    | 2.9M   |
|         |       | Transformer| 0.7902 | 0.3663 | 0.7939 | 0.6169    | 2.5M   |
| KIBA    | Random| Residual   | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         |       | Transformer| x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | Residual   | 0.4135 | 0.3186 | 0.7250 | 0.5820    | 34.1M  |
|         |       | Transformer| 0.4295 | 0.2921 | 0.7172 | 0.5607    | 27.2M  |

## Attentive vs. concat-based aggregator

For multi-modal model only:

| Dataset | Split | Aggregation | MSE ↓  | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|---------|-------|------------|--------|--------|--------|-----------|--------|
| Davis   | Random| Concat     | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         |       | Attentive  | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | Concat     | 0.7713 | 0.3814 | 0.7998 | 0.6319    | 2.7M   |
|         |       | Attentive  | 0.7560 | 0.3937 | 0.7997 | 0.6460    | 3.7M   |
| KIBA    | Random| Concat     | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         |       | Attentive  | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | Concat     | 0.4362 | 0.2811 | 0.7154 | 0.5527    | 27.9M  |
|         |       | Attentive  | 0.4120 | 0.3210 | 0.7243 | 0.5824    | 31.1M  |

## Fingerprint vs. Embeddings

| Dataset | Split | Input | MSE ↓  | R2 ↑   | CI ↑   | Pearson ↑ | Params |
|---------|-------|------------|--------|--------|--------|-----------|--------|
| Davis   | Random| FP         | 0.2352 | 0.6924 | 0.8896 | 0.8324    | 13.9M  |
|         |       | EMB        | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | FP         | 0.8609 | 0.3096 | 0.7664 | 0.5677    |  1.7M  |
|         |       | EMB        | 0.7538 | 0.3955 | 0.8003 | 0.6437    | 2.9M   |
| KIBA    | Random| FP         | 0.1573 | 0.7648 | 0.8595 | 0.8749    | 16.9M  |
|         |       | EMB        | x.xxxx | x.xxxx | x.xxxx | x.xxxx    | xx.xM  |
|         | Cold  | FP         | 0.3702 | 0.3899 | 0.7335 | 0.6481    | 15.4M  |
|         |       | EMB        | 0.4120 | 0.3210 | 0.7243 | 0.5824    | 31.1M  |

### Fingerprint vs. Embeddings on combined dataset

| Split | Input | MSE pKd ↓ | MSE pKi ↓ | MSE KIBA ↓ | Accuracy ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ | Params |
|-------|-------|-----------|-----------|------------|------------|------|---------|---------|--------|
| Random| FP    | 0.2352    | 0.6633    | 0.2358    | 0.8742    | 0.7679 | 0.9253 | 0.8451 | 44.3M |
|       | EMB   | 0.1573    | 0.7014    | 0.2104    | 0.8627    | 0.7556 | 0.9225 | 0.8320 | 65.9M |
| Cold  | FP    | 0.8609    | 0.8288    | 0.4535    | 0.8458    | 0.6929 | 0.8816 | 0.7713 | 45.1M |
|       | EMB   | 0.3702    | 0.8862    | 0.4701    | 0.8300    | 0.6785 | 0.8743 | 0.7446 | 65.9M |


### Most informative embedding (in models supporting attentive aggregator)

### Single embedding vs. multi-embedding vs. fingerprints

