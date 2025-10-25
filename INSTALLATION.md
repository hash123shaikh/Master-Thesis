# Installation Guide

Complete step-by-step installation guide for the Multimodal Cancer Survival Prediction pipeline.

---

## Prerequisites

### System Requirements
- **OS**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 5GB free space
- **GPU** (optional but recommended): NVIDIA GPU with CUDA support

### Software Requirements
- Python 3.8 or higher
- pip (Python package manager)
- Git

---

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/multimodal-cancer-prediction.git
cd multimodal-cancer-prediction
```

### 2. Set Up Python Environment

**Option A: Using venv (Recommended)**

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Option B: Using Conda**

```bash
conda create -n cancer_pred python=3.8
conda activate cancer_pred
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**If you encounter errors:**

- **TensorFlow installation issues**: Try `pip install tensorflow-cpu==2.8.0` for CPU-only version
- **NumPy compatibility**: Ensure version 1.23.5 or compatible with your TensorFlow
- **Windows long path errors**: Enable long paths in registry or use shorter installation directory

### 4. Verify Installation

```bash
python test_installation.py
```

Expected output:
```
============================================================
Checking Installation for Multimodal Cancer Prediction
============================================================

✓ TensorFlow            2.8.0
✓ NumPy                 1.23.5
✓ Pandas                1.5.3
✓ scikit-learn          1.2.2
✓ Matplotlib            3.7.1

============================================================
✓ All required packages are installed!
...
✓ Installation successful! You're ready to run the pipeline.
```

---

## Data Setup

### 1. Download METABRIC Dataset

1. Visit [cBioPortal METABRIC Study](https://www.cbioportal.org/study/summary?id=brca_metabric)
2. Click "Download" → Select data types:
   - Clinical data
   - Gene expression (mRNA)
   - Copy number alterations (CNA)
3. Save files to `code/Data/METABRIC/` directory

### 2. Organize Data Files

Your directory structure should look like:

```
Master-Thesis-Work/
├── code/
│   ├── Data/
│   │   └── METABRIC/
│   │       ├── METABRIC_clinical_1980.txt
│   │       ├── METABRIC_cnv_[date].txt
│   │       └── METABRIC_expr_[date].txt
│   ├── GaAtCNN_cln.py
│   ├── GaAtCNN_cnv.py
│   ├── GaAtCNN_expr.py
│   └── RF.py
```

### 3. Update File Paths in Scripts

**Critical Step**: Update hardcoded paths in each Python file:

#### In `code/GaAtCNN_cln.py`:

Find (around line 37):
```python
dataset_clinical = numpy.loadtxt("F:/Dissertations/.../METABRIC_clinical_1980.txt", delimiter="\t")
```

Replace with your path:
```python
dataset_clinical = numpy.loadtxt("code/Data/METABRIC/METABRIC_clinical_1980.txt", delimiter="\t")
```

Also update (around line 28):
```python
path = 'YOUR_PATH/gatedAtnClnOutput.csv'
```

#### In `code/GaAtCNN_cnv.py`:

Update path to CNV data file (similar to above)

#### In `code/GaAtCNN_expr.py`:

Update path to expression data file

#### In `code/RF.py`:

Find (around line 28):
```python
path = "F:/Dissertations/.../Data/METABRIC"
file1 = "gatedAtnAll_Input.csv"
```

Replace with:
```python
path = "results"  # Directory where Stage 1 outputs are saved
file1 = "gatedAtnAll_Input.csv"
```

---

## Troubleshooting

### Common Installation Issues

#### 1. TensorFlow GPU Not Detected

**Symptom**: `tf.test.is_gpu_available()` returns `False`

**Solution**:
- Install CUDA Toolkit 11.2 and cuDNN 8.1
- Verify GPU: `nvidia-smi`
- For CPU-only: Use `tensorflow-cpu` package (slower but works)

#### 2. ImportError: DLL load failed

**Platform**: Windows

**Solution**:
```bash
pip install --upgrade tensorflow
# Or install Microsoft Visual C++ Redistributable
```

#### 3. NumPy/Pandas Compatibility Issues

**Solution**:
```bash
pip uninstall numpy pandas
pip install numpy==1.23.5 pandas==1.5.3
```

#### 4. Memory Errors During Training

**Symptom**: `ResourceExhaustedError` or system freezing

**Solution**:
- Reduce batch size in scripts (change `batch_size=8` to `batch_size=4`)
- Close other applications
- Use CPU-only TensorFlow if GPU memory is insufficient

#### 5. Path Not Found Errors

**Symptom**: `FileNotFoundError: [Errno 2] No such file or directory`

**Solution**:
- Double-check data file paths in each script
- Use absolute paths instead of relative if issues persist
- Ensure directory separators match your OS (`/` for Linux/Mac, `\` for Windows)

---

## Platform-Specific Notes

### Windows

- Use forward slashes `/` in paths for cross-platform compatibility
- If using Anaconda, activate environment before installation
- Long path issues: Run as Administrator if needed

### macOS

- If using M1/M2 Mac: Use `tensorflow-macos` and `tensorflow-metal`
```bash
pip install tensorflow-macos==2.8.0
pip install tensorflow-metal
```

### Linux

- Install system dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-dev python3-pip
```

- For GPU support:
```bash
# Install CUDA and cuDNN from NVIDIA
# Then install TensorFlow with GPU
pip install tensorflow-gpu==2.8.0
```

---

## Next Steps

After successful installation:

1. **Test with small dataset**: Run on subset of data first
2. **Verify outputs**: Check that CSV files are generated in `results/`
3. **Monitor resource usage**: Use Task Manager/Activity Monitor
4. **Read main README**: See [README.md](README.md) for full usage instructions

---

## Getting Help

If you encounter issues not covered here:

1. Check [README.md](README.md) FAQ section
2. Search GitHub Issues
3. Create new issue with:
   - Python version: `python --version`
   - Package versions: `pip list`
   - Error message (full traceback)
   - Operating system

---

## Uninstallation

To remove the environment:

```bash
# For venv:
deactivate
rm -rf venv/

# For conda:
conda deactivate
conda env remove -n cancer_pred
```

---

Last updated: [26th October 2025]
