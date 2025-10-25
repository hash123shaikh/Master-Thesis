# 🧬 Gated Attention CNN for Multimodal Cancer Survival Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8.0-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Thesis](https://img.shields.io/badge/Type-Master's%20Thesis-purple.svg)]()

> **Master's Thesis Research** | Department of Computer Engineering, Aligarh Muslim University  
> **Author**: Hasan Shaikh | **Supervisor**: Prof. Rashid Ali

---

## 📋 Table of Contents
- [Overview](#overview)
- [Research Contributions](#research-contributions)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Current Limitations](#current-limitations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Contact](#contact)

---

## 🎯 Overview

This repository implements a **two-stage deep learning pipeline** for breast cancer survival prediction using the METABRIC dataset. The approach combines:

1. **Stage 1**: Three modality-specific Gated Attention CNNs for feature extraction
2. **Stage 2**: Random Forest ensemble for final survival classification

### Research Problem

Traditional survival prediction models face challenges with:
- Integration of heterogeneous multimodal medical data (clinical, genomic, transcriptomic)
- High-dimensional feature spaces with complex non-linear relationships
- Capturing complementary information across different biological data types

### Our Approach

**Two-Stage Pipeline:**

```
┌──────────────────┐
│  Clinical Data   │──→ Gated Attention CNN ──→ 50 features ┐
│  (25 features)   │                                         │
└──────────────────┘                                         │
                                                             ├──→ Concatenate ──→ Random Forest ──→ Prediction
┌──────────────────┐                                         │      (775 features)    (200 trees)
│ Gene Expression  │──→ Gated Attention CNN ──→ 525 features│
│  (XXX features)  │                                         │
└──────────────────┘                                         │
                                                             │
┌──────────────────┐                                         │
│   CNA Data       │──→ Gated Attention CNN ──→ 200 features┘
│  (XXX features)  │
└──────────────────┘
```

**Key Innovation**: Gated attention mechanism adaptively weights features within each modality before ensemble fusion.

---

## 🌟 Research Contributions

| Contribution | Description |
|-------------|-------------|
| **Modality-Specific Architecture** | Independent Gated Attention CNNs for clinical, gene expression, and copy number alteration data |
| **Multi-Scale Feature Extraction** | Parallel convolutional branches with different kernel sizes (1, 2, 3) |
| **Hierarchical Fusion** | Two-stage approach: CNN-based feature learning + ensemble-based integration |
| **Comprehensive Evaluation** | 10-fold stratified cross-validation on METABRIC dataset (1,980 patients) |

---

## 🔬 Methodology

### Stage 1: Modality-Specific Gated Attention CNNs

Each modality is processed independently through a specialized architecture:

**Architecture Components:**
- **Multi-Branch Convolution**: Parallel Conv1D layers with kernel sizes 1 and 2
- **Gated Attention Mechanism**: Element-wise multiplication between gating signal and feature maps
- **Max Pooling**: Dimensionality reduction while preserving salient features
- **Dense Layers**: Sequential layers (150 → 100 → 50 neurons) for feature abstraction

**Gating Mechanism:**
```
Conv1D(input) → [Gate1(k=1), Gate2(k=3)] → [Multiply, Multiply] → MaxPool → Concatenate
```

### Stage 2: Random Forest Ensemble

- **Input**: Concatenated features from all three modalities (775 dimensions)
- **Algorithm**: Random Forest with 200 trees
- **Strategy**: Balanced class weights to handle survival class imbalance
- **Validation**: 10-fold stratified cross-validation

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| CNN Filters | 25 per branch |
| Epochs | 25 |
| Batch Size | 8 |
| Dropout Rate | 0.25 |
| L2 Regularization | 0.001 |
| Specificity Target | 0.95 |
| RF Estimators | 200 |
| CV Folds | 10 |

---

## 📊 Dataset

**METABRIC (Molecular Taxonomy of Breast Cancer International Consortium)**

| Attribute | Details |
|-----------|---------|
| **Total Patients** | 1,980 |
| **Data Modalities** | 3 (Clinical, Gene Expression, Copy Number Alteration) |
| **Outcome** | Binary survival (≥5 years vs <5 years) |
| **Clinical Features** | 25 (age, tumor size, grade, ER/PR/HER2 status, etc.) |
| **Gene Expression** | Discretized mRNA expression levels |
| **CNA Features** | Copy number alteration profiles |
| **Source** | [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric) |

**Preprocessing:**
- Missing value imputation (median strategy)
- Feature standardization for clinical variables
- Gene expression discretization: {-1: under-expressed, 0: normal, 1: over-expressed}
- 10-fold stratified cross-validation to maintain class distribution

---

## 📈 Results

### Performance Summary

**Final Ensemble Model (10-Fold CV):**

| Metric | Value | Standard Deviation |
|--------|-------|--------------------|
| **Accuracy** | **91.2%** | ±X.X% |
| **AUC** | **0.950** | ±0.XXX |
| **Sensitivity** | XX.X% | ±X.X% |
| **Specificity** | XX.X% | ±X.X% |
| **Precision** | 84.1% | ±X.X% |

### Comparison with Published Methods

| Method | Dataset | Accuracy | AUC | Reference |
|--------|---------|----------|-----|-----------|
| **This Work (RF Ensemble)** | METABRIC | **91.2%** | **0.950** | — |
| Stacked RF | METABRIC | 90.2% | 0.930 | Arya & Saha (2022) |
| MDNNMD | METABRIC | 82.6% | 0.845 | Sun et al. (2018) |
| Random Forest | METABRIC | 79.1% | 0.801 | Baseline |
| SVM | METABRIC | 80.5% | 0.810 | Baseline |
| Logistic Regression | METABRIC | 76.0% | 0.663 | Baseline |

*Note: Results from literature may use different train/test splits and preprocessing*

### Individual Modality Performance

| Modality | Features (Input → Output) | AUC | Notes |
|----------|---------------------------|-----|-------|
| Clinical | 25 → 50 | 0.XXX | Strong baseline performance |
| Gene Expression | XXX → 525 | 0.XXX | High-dimensional feature learning |
| CNA | XXX → 200 | 0.XXX | Genomic alteration patterns |
| **Combined (All)** | **775** | **0.950** | **Best performance** |

**Key Finding**: Multimodal integration provides superior predictive performance compared to any single modality.

---

## 🚀 Installation

### Prerequisites

```bash
Python >= 3.8
TensorFlow >= 2.8.0
scikit-learn >= 1.2.0
numpy >= 1.23.0
pandas >= 1.5.0
matplotlib >= 3.7.0
```

### Setup Instructions

**Step 1: Clone Repository**
```bash
git clone https://github.com/yourusername/multimodal-cancer-prediction.git
cd multimodal-cancer-prediction
```

**Step 2: Create Virtual Environment**
```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

**Step 3: Install Dependencies**

Option A - Using requirements.txt:
```bash
pip install -r requirements.txt
```

Option B - Manual installation:
```bash
pip install tensorflow==2.8.0
pip install scikit-learn==1.2.2
pip install numpy==1.23.5
pip install pandas==1.5.3
pip install matplotlib==3.7.1
```

**Step 4: Verify Installation**
```bash
python test_installation.py
```

Expected output:
```
✓ TensorFlow 2.8.0
✓ scikit-learn 1.2.2
✓ NumPy 1.23.5
✓ Installation successful!
```

---

## 💻 Usage

### ⚠️ Important: Update File Paths First

Before running any script, update the hardcoded file paths to match your system:

**In `code/GaAtCNN_cln.py` (line ~37):**
```python
# Change this:
dataset_clinical = numpy.loadtxt("F:/Dissertations/.../METABRIC_clinical_1980.txt", delimiter="\t")

# To your path:
dataset_clinical = numpy.loadtxt("/your/path/to/Data/METABRIC/METABRIC_clinical_1980.txt", delimiter="\t")
```

**Repeat for:**
- `code/GaAtCNN_cnv.py` (update path variable)
- `code/GaAtCNN_expr.py` (update path variable)
- `code/RF.py` (update path variable, line ~28)

### Running the Complete Pipeline

The pipeline requires running **4 separate scripts in sequence**:

#### **Stage 1: Train Individual Modality Models**

```bash
# 1. Train Clinical Gated Attention CNN
python code/GaAtCNN_cln.py
# Output: results/gatedAtnClnOutput.csv (50 clinical features)

# 2. Train CNV Gated Attention CNN
python code/GaAtCNN_cnv.py
# Output: results/gatedAtnCnvOutput.csv (200 CNV features)

# 3. Train Gene Expression Gated Attention CNN
python code/GaAtCNN_expr.py
# Output: results/gatedAtnExprOutput.csv (525 expression features)
```

**What Happens:**
- Each script trains a CNN on its respective modality using 10-fold CV
- Models extract learned features from the penultimate dense layer (50/200/525 dimensions)
- Features are saved to CSV files for Stage 2

#### **Stage 2: Train Random Forest Ensemble**

```bash
# 4. Combine features and train ensemble
python code/RF.py
# Reads: gatedAtnClnOutput.csv, gatedAtnCnvOutput.csv, gatedAtnExprOutput.csv
# Trains: Random Forest on combined 775 features
# Outputs: Final predictions and ROC curves
```

### Expected Runtime

| Script | Approx. Time | Output |
|--------|--------------|--------|
| GaAtCNN_cln.py | ~15-20 min | Clinical features (50 dims) |
| GaAtCNN_cnv.py | ~20-30 min | CNV features (200 dims) |
| GaAtCNN_expr.py | ~30-45 min | Expression features (525 dims) |
| RF.py | ~5-10 min | Final predictions + metrics |
| **Total** | **~70-105 min** | Complete pipeline |

*Runtime depends on hardware (GPU availability)*

### Understanding the Output

After running all scripts, you'll have:

```
results/
├── gatedAtnClnOutput.csv          # Clinical features (1980 × 50)
├── gatedAtnCnvOutput.csv          # CNV features (1980 × 200)
├── gatedAtnExprOutput.csv         # Expression features (1980 × 525)
├── clinical_gated_attention.png   # Model architecture diagram
├── roc_curve_clinical.png         # Clinical CNN ROC
├── roc_curve_cnv.png              # CNV CNN ROC
├── roc_curve_expression.png       # Expression CNN ROC
└── roc_curve_ensemble.png         # Final ensemble ROC
```

---

## ⚠️ Current Limitations

This is a **thesis research implementation** with the following constraints:

### Research Limitations

- **Single Dataset**: Evaluated only on METABRIC (requires validation on other cohorts)
- **Binary Classification**: Only 5-year survival cutoff (could extend to time-to-event)
- **No Explainability**: Predictions are not interpretable (SHAP analysis needed)
- **Preprocessing Not Included**: Assumes data is already preprocessed

### Reproducibility Notes

- **Random Seeds**: Set to 1 in all scripts for reproducibility
- **Hardware Variance**: Results may vary slightly due to GPU non-determinism
- **Data Splits**: 10-fold CV ensures robust evaluation, but exact folds differ from published work

---

## 📁 Project Structure

```
Master-Thesis-Work/
├── code/
│   ├── Data/
│   │   └── METABRIC/
│   │       ├── METABRIC_clinical_1980.txt    # Clinical data
│   │       ├── METABRIC_cnv_XXXX.txt         # Copy number alteration
│   │       └── METABRIC_expr_XXXX.txt        # Gene expression
│   ├── GaAtCNN_cln.py          # Stage 1: Clinical CNN (outputs 50 features)
│   ├── GaAtCNN_cnv.py          # Stage 1: CNV CNN (outputs 200 features)
│   ├── GaAtCNN_expr.py         # Stage 1: Expression CNN (outputs 525 features)
│   └── RF.py                   # Stage 2: Random Forest ensemble
├── docs/
│   ├── Hasan_MTech_Dissertation_PPT.pdf      # Thesis presentation
│   └── Hasan_Dissertation_Report.pdf         # Full thesis document
├── results/                    # Generated after running scripts (not in repo)
│   ├── gatedAtnClnOutput.csv   # Clinical features (created by GaAtCNN_cln.py)
│   ├── gatedAtnCnvOutput.csv   # CNV features (created by GaAtCNN_cnv.py)
│   ├── gatedAtnExprOutput.csv  # Expression features (created by GaAtCNN_expr.py)
│   └── figures/                # ROC curves and model diagrams
├── requirements.txt            # Python dependencies
├── test_installation.py        # Verify environment setup
├── .gitignore                  # Ignore generated files
├── LICENSE                     # MIT License
└── README.md                   # This file
```

### Key Files Explained

| File | Purpose | Input | Output |
|------|---------|-------|--------|
| `GaAtCNN_cln.py` | Train clinical CNN | `METABRIC_clinical_1980.txt` | `gatedAtnClnOutput.csv` (50 features) |
| `GaAtCNN_cnv.py` | Train CNV CNN | `METABRIC_cnv_XXXX.txt` | `gatedAtnCnvOutput.csv` (200 features) |
| `GaAtCNN_expr.py` | Train expression CNN | `METABRIC_expr_XXXX.txt` | `gatedAtnExprOutput.csv` (525 features) |
| `RF.py` | Train RF ensemble | All 3 CSVs above | Final predictions + ROC curves |

---

## 📝 Citation

### Thesis

If you use this work, please cite:

```bibtex
@mastersthesis{shaikh2023multimodal,
  title={Multimodal Data Analytics for Predicting the Survival of Cancer Patients},
  author={Shaikh, Hasan},
  year={2023},
  school={Aligarh Muslim University},
  type={Master's Thesis},
  department={Computer Engineering},
  supervisor={Ali, Rashid}
}
```

### Related Publications

This work builds upon:

1. **Sun, D., et al.** (2018). MDNNMD: A deep neural network with multidimensional data for survival prediction. *BMC Bioinformatics*, 19(1), 1-13.

2. **Arya, N., & Saha, S.** (2022). Multi-modal classification for human breast cancer prognosis prediction. *Scientific Reports*, 12(1), 1-13.

3. **Curtis, C., et al.** (2012). The genomic and transcriptomic architecture of 2,000 breast tumours reveals novel subgroups. *Nature*, 486(7403), 346-352.

---

## 📧 Contact

**Hasan Shaikh**  
M.Tech Student, Computer Engineering  
Aligarh Muslim University, India

📧 Email: [hasanshaikh3198@gmail.com](mailto:hasanshaikh3198@gmail.com)  
🔗 LinkedIn: [https://linkedin.com/in/hasann-shaikh](https://linkedin.com/in/hasann-shaikh)  
🐙 GitHub: [https://github.com/hash123shaikh](https://github.com/hash123shaikh)

**Supervisor:**  
Prof. Rashid Ali  
Department of Computer Engineering, AMU  
📧 [rashidali@zhcet.ac.in](mailto:rashidali@zhcet.ac.in)

---

## 🙏 Acknowledgments

- **METABRIC Consortium** for making the dataset publicly available
- **cBioPortal** for data hosting and access infrastructure
- **Department of Computer Engineering, AMU** for computational resources
- **Prof. Rashid Ali** for thesis supervision and guidance
- Open-source communities (TensorFlow, scikit-learn) for excellent tools

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Academic Use**: Free to use for research and education  
**Commercial Use**: Contact author for licensing arrangements  
**Attribution**: Please cite the thesis if you use this code

---

## 🐛 Known Issues & FAQs

### Common Problems

**Q: "FileNotFoundError: [Errno 2] No such file or directory"**  
**A:** Update the hardcoded path in the script to match your data location (see [Usage](#usage) section)

**Q: "ValueError: could not convert string to float"**  
**A:** Check your data file format. Ensure delimiter is `\t` (tab) for clinical data, `,` (comma) for RF.py

**Q: "ModuleNotFoundError: No module named 'tensorflow'"**  
**A:** Ensure virtual environment is activated and TensorFlow is installed: `pip install tensorflow==2.8.0`

**Q: Scripts run but no output files generated**  
**A:** Check the `path` variable in each script - it must point to a writable directory

**Q: Different results than reported**  
**A:** Slight variance is normal due to GPU non-determinism. Run multiple times and average results.

### Getting Help

1. **Check this README** for common solutions
2. **Review thesis document** (`docs/Hasan_Dissertation_Report.pdf`) for methodology details
3. **Open GitHub issue** with error message and environment details
4. **Email author** for complex problems

---

## 🔗 Additional Resources

- **METABRIC Dataset**: [cBioPortal Study Page](https://www.cbioportal.org/study/summary?id=brca_metabric)
- **TensorFlow Documentation**: [tensorflow.org](https://www.tensorflow.org/)
- **scikit-learn User Guide**: [scikit-learn.org](https://scikit-learn.org/stable/user_guide.html)
- **Thesis Full Text**: See `docs/Hasan_Dissertation_Report.pdf`

---

<div align="center">
  <sub>Built for advancing cancer research through deep learning</sub><br>
  <sub>Master's Thesis Project | Aligarh Muslim University | 2023</sub>
</div>
