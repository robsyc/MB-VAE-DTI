# Single-score DTI benchmark results

- Illiadis: https://doi.org/10.1186/s12859-024-05684-y
- TxGemma: https://arxiv.org/abs/2504.06196
- SSM-DTA: https://doi.org/10.1093/bib/bbad386 (very similar goals as our thesis; they note on pretraining strategies)
- MMELON (IBM Biomed): https://arxiv.org/abs/2410.19704

Baseline & multi-modal model results were more extensively tuned and the average test-set results over the 5 best configurations were reported.The other models, due to pre-training, were not tuned as extensively and the results report the best test-set results of the single best configuration (based on the validation set).

## DAVIS (pKd)

### Random Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.2352 | 0.4849 | 0.6924 | 0.8896 | 0.8324 | 13.9M |
| Multi-modal | 0.240136 | 0.489898 | 0.685882 | 0.8806 | 0.8286 | 7.7M | **REDO**
| Multi-output | 0.2585 | 0.5085 | 0.6941 | 0.8776 | 0.8414 | 44.3M |
| Multi-hybrid | 0.2153 | 0.4640 | 0.7384 | 0.9001 | 0.8702 | 65.9M |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.2656 | | 0.6733 | 0.8702 | | 2.1M |
| SSM-DTA | 0.219 | | | 0.890 | | |
| MMELON | | 0.44 | | | | 84M |


### Cold Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.8609 | 0.9274 | 0.3096 | 0.7664 | 0.5677 | 1.7M |
| Multi-modal | 0.7538 | 0.8681 | 0.3955 | 0.8003 | 0.6437 | 2.9M |
| Multi-output | 0.9612 | 0.9804 | 0.3861 | 0.7561 | 0.6510 | 45.1M |
| Multi-hybrid | 0.8860 | 0.9413 | 0.2895 | 0.7346 | 0.5858 | 65.9M |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.6189 | |  -0.0426 | 0.7304 | | 6.3M |
| TxGemma (27B-Predict) | 0.555 | | | | | 27B |
| SSM-DTA | 0.8019 | | 0.2803 | | | |

## KIBA (KIBA score)

### Random Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.1573 | 0.3966 | 0.7648 | 0.8595 | 0.8749 | 16.9M |
| Multi-modal | 0.195592 | 0.442218 | 0.707488 | 0.84797 | 0.841234 | 10.1M | **REDO**
| Multi-output | 0.2358 | 0.4856 | 0.6475 | 0.8389 | 0.8213 | 44.3M |
| Multi-hybrid | 0.2103 | 0.4587 | 0.6853 | 0.8618 | 0.8536 | 65.9M |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.1994 | |  0.7187 | 0.8379 | | 2.1M |
| SSM-DTA | 0.154 | | | 0.895 | | |

### Cold Split

| Model | MSE ↓ | RMSE ↓ | R2 ↑ | CI ↑ | Pearson ↑ | Params |
|-------|-----|------|----|----|---------|--------|
| Baseline | 0.3702 | 0.6084 | 0.3899 | 0.7335 | 0.6481 | 15.4M |
| Multi-modal | 0.427438 | 0.653584 | 0.29555 | 0.709544 | 0.550116 | 18.6M | **REDO**
| Multi-output | 0.4535 | 0.6734 | 0.2525 | 0.7171 | 0.5808 | 45.1M |
| Multi-hybrid | 0.4701 | 0.6856 | 0.2252 | 0.7374 | 0.5883 | 65.9M |
| Full | | | | | |  |
|-------|-----|------|----|----|---------|--------|
| Iliadis (MLP-MLP) | 0.4167 | |  0.4498 | 0.7510 | | 9.9M |
| TxGemma (9B-Predict) | 0.588 | | | | | 9B |


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

## Attentive vs. concat-based aggregator