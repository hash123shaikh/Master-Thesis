# Complete Results Summary

## Master's Thesis: Multimodal Cancer Survival Prediction
**Author**: Hasan Shaikh | **Institution**: Aligarh Muslim University

---

## Table 1: Individual Modality Performance (Ablation Study)

### Accuracy by Modality
| Modality | Accuracy | Rank |
|----------|----------|------|
| Clinical Only | 81.3% | 3rd |
| Gene Expression Only | 84.1% | 2nd |
| CNA Only | 89.3% | 1st |
| **Multimodal (All Combined)** | **91.2%** | **Best** |

### Precision by Modality
| Modality | Precision | Rank |
|----------|-----------|------|
| Clinical Only | 71.2% | 3rd |
| Gene Expression Only | 77.9% | 2nd |
| CNA Only | 84.1% | 1st (tied with multimodal) |
| **Multimodal (All Combined)** | **84.1%** | **1st (tied)** |

### Sensitivity by Modality
| Modality | Sensitivity | Rank |
|----------|-------------|------|
| Clinical Only | 41.3% | 3rd |
| Gene Expression Only | 50.5% | 2nd |
| CNA Only | 70.2% | 1st |
| **Multimodal (All Combined)** | **79.8%** | **Best (+9.6%)** |

### AUC by Modality
| Modality | AUC | Rank |
|----------|-----|------|
| Clinical Only | 0.834 | 3rd |
| CNA Only | 0.850 | 2nd |
| Gene Expression Only | 0.923 | 1st |
| **Multimodal (All Combined)** | **0.950** | **Best (+0.027)** |

---

## Table 2: Comparison with State-of-the-Art Methods

### Accuracy Comparison (METABRIC Dataset)
| Rank | Method | Accuracy | Gap from Our Method |
|------|--------|----------|---------------------|
| 1 | **Our Method (Multimodal GaAtCNN)** | **91.2%** | — |
| 2 | Stacked RF [12] | 90.2% | -1.0% |
| 3 | CNA Only (Our ablation) | 89.3% | -1.9% |
| 4 | MDNNMD [18] | 82.6% | -8.6% |
| 5 | Support Vector Machine [21] | 80.5% | -10.7% |
| 6 | Random Forest [20] | 79.1% | -12.1% |
| 7 | Logistic Regression [19] | 76.0% | -15.2% |

### Complete Metrics Comparison
| Method | Accuracy | Precision | Sensitivity | AUC |
|--------|----------|-----------|-------------|-----|
| **Multimodal GaAtCNN (Ours)** | **91.2%** | **84.1%** | **79.8%** | **0.950** |
| Stacked RF [12] | 90.2% | 84.1% | 74.7% | 0.930 |
| MDNNMD [18] | 82.6% | 74.9% | 45.0% | 0.845 |
| Support Vector Machine [21] | 80.5% | 70.8% | 36.5% | 0.810 |
| Random Forest [20] | 79.1% | 76.6% | 22.6% | 0.801 |
| Logistic Regression [19] | 76.0% | 54.9% | 18.3% | 0.663 |

---

## Table 3: Key Performance Improvements

### Improvement Over Best Single Modality
| Metric | Single Best | Multimodal | Absolute Gain | Relative Gain |
|--------|-------------|------------|---------------|---------------|
| Accuracy | 89.3% (CNA) | 91.2% | +1.9% | +2.1% |
| Precision | 84.1% (CNA) | 84.1% | 0.0% | 0.0% |
| Sensitivity | 70.2% (CNA) | 79.8% | +9.6% | +13.7% |
| AUC | 0.923 (Gene Expr) | 0.950 | +0.027 | +2.9% |

### Improvement Over Traditional Machine Learning
| Baseline Method | Accuracy Gain | Sensitivity Gain | AUC Gain |
|-----------------|---------------|------------------|----------|
| Logistic Regression | +15.2% | +61.5% | +0.287 |
| Random Forest | +12.1% | +57.2% | +0.149 |
| SVM | +10.7% | +43.3% | +0.140 |

### Improvement Over Deep Learning Baselines
| Deep Learning Method | Accuracy Gain | Sensitivity Gain | AUC Gain |
|---------------------|---------------|------------------|----------|
| MDNNMD [18] | +8.6% | +34.8% | +0.105 |

---

## Table 4: Statistical Significance

### Metric Variance Across Modalities

| Metric | Min (Clinical) | Max (Multimodal) | Range |
|--------|----------------|------------------|-------|
| Accuracy | 81.3% | 91.2% | 9.9% |
| Precision | 71.2% | 84.1% | 12.9% |
| Sensitivity | 41.3% | 79.8% | 38.5% |
| AUC | 0.834 | 0.950 | 0.116 |

**Notable**: Sensitivity shows the largest improvement from unimodal to multimodal (+38.5% range), indicating the multimodal approach is particularly effective at identifying positive cases (patients with poor survival).

---

## Summary Statistics

### Overall Performance Profile
- **Best Metric**: AUC (0.950) - Excellent discrimination
- **Most Improved Metric**: Sensitivity (79.8% vs 41.3% for clinical only)
- **Competitive Advantage**: +1.0% accuracy over nearest competitor
- **Consistency**: Top-ranked across all 4 metrics

### Clinical Implications
- **High Sensitivity (79.8%)**: Identifies ~80% of patients at risk
- **High Precision (84.1%)**: 84% of predicted high-risk patients are correct
- **Excellent AUC (0.950)**: Strong discrimination for clinical decision-making
- **Balanced Performance**: No single metric is weak

---

## Feature Extraction Summary

### Learned Feature Dimensions
| Modality | Input Features | Extracted Features | Compression Ratio |
|----------|----------------|-------------------|-------------------|
| Clinical | 25 | 50 | 2.0× expansion |
| Gene Expression | ~400 (exact unclear) | 525 | ~1.3× expansion |
| CNA | ~200 (exact unclear) | 200 | 1.0× maintained |
| **Total Combined** | **~625** | **775** | **~1.2× overall** |

---

## References for Compared Methods

[12] Arya, N., & Saha, S. (2022). Stacked ensemble approach for breast cancer prognosis.  
[18] Sun, D., et al. (2018). MDNNMD: Multi-dimensional neural network for multimodal data.  
[19] Baseline: Logistic Regression implementation  
[20] Baseline: Random Forest implementation  
[21] Baseline: Support Vector Machine implementation

---

## Recommended Figures for Presentation

1. **Bar chart**: Accuracy comparison across all methods
2. **Grouped bar chart**: All 4 metrics (Accuracy, Precision, Sensitivity, AUC) by method
3. **ROC curves**: Overlay of unimodal vs multimodal
4. **Heatmap**: Confusion matrices for each modality
5. **Spider/Radar plot**: Multi-metric comparison showing balance

---

**Document Date**: Based on thesis results  
**Dataset**: METABRIC (1,980 patients)  
**Validation**: 10-fold stratified cross-validation  
**Statistical Test**: Results reported as point estimates (standard deviations not provided)
