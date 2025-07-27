# Single-score DTI benchmark results

- Illiadis: https://doi.org/10.1186/s12859-024-05684-y
- TxGemma: https://arxiv.org/abs/2504.06196
- SSM-DTA: https://doi.org/10.1093/bib/bbad386 (very similar goals as our thesis; they note on pretraining strategies)
- MMELON (IBM Biomed): https://arxiv.org/abs/2410.19704

## DAVIS (pKd)

### Random Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.23359 | 0.483302 | 0.694448 | 0.891916 | 0.833774 | 15.7M |
| Multi-modal | 0.240136 | 0.489898 | 0.685882 | 0.88058 | 0.828586 | 7.7M |
| Multi-output | | | | | |  |
| Multi-hybrid | | | | | |  |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.2656 | | 0.6733 | 0.8702 | | 2.1M |
| SSM-DTA | 0.219 | | | 0.890 | | |
| MMELON | | 0.44 | | | | 84M |


### Cold Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.834226 | 0.913028 | 0.330966 | 0.780424 | 0.588732 | 1.6M |
| Multi-modal | 0.858424 | 0.926012 | 0.311564 | 0.763706 | 0.570308 | 9.5M |
| Multi-output | | | | | |  |
| Multi-hybrid | | | | | |  |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.6189 | |  -0.0426 | 0.7304 | | 6.3M |
| TxGemma (27B-Predict) | 0.555 | | | | | 27B |
| SSM-DTA | 0.8019 | | 0.2803 | | | |

## KIBA (pKi)

### Random Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.159508 | 0.399382 | 0.761456 | 0.859122 | 0.87306 | 19.7M |
| Multi-modal | | | | | | 10.1M |
| Multi-output | | | | | |  |
| Multi-hybrid | | | | | |  |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.1994 | |  0.7187 | 0.8379 | | 2.1M |
| SSM-DTA | 0.154 | | | 0.895 | | |

### Cold Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.375726 | 0.612934 | 0.38079 | 0.733286 | 0.642546 | 11.9M |
| Multi-modal | 0.427438 | 0.653584 | 0.29555 | 0.709544 | 0.550116 | 18.6M |
| Multi-output | | | | | |  |
| Multi-hybrid | | | | | |  |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.4167 | |  0.4498 | 0.7510 | | 9.9M |
| TxGemma (9B-Predict) | 0.588 | | | | | 9B |


# Multi-score DTI results

### Random Split

| Model | MSE pKd ↓ | MSE pKi ↓ | MSE KIBA ↓ | Accuracy ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ |
| Multi-output | | | | | | | |
| Multi-hybrid | | | | | | | |
| Full | | | | | | | |

### Cold Split

| Model | MSE pKd ↓ | MSE pKi ↓ | MSE KIBA ↓ | Accuracy ↑ | F1 ↑ | AUROC ↑ | AUPRC ↑ |
| Multi-output | | | | | | | |
| Multi-hybrid | | | | | | | |
| Full | | | | | | | |
