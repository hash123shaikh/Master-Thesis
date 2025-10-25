#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test installation of required packages for Multimodal Cancer Prediction.

Run this script after installation to verify all dependencies are correctly installed.
"""

import sys

def check_package(package_name, import_name=None):
    """Check if a package is installed and print version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {package_name:20s} {version}")
        return True
    except ImportError:
        print(f"✗ {package_name:20s} NOT INSTALLED")
        return False

def main():
    print("=" * 60)
    print("Checking Installation for Multimodal Cancer Prediction")
    print("=" * 60)
    print()
    
    required_packages = [
        ('TensorFlow', 'tensorflow'),
        ('NumPy', 'numpy'),
        ('Pandas', 'pandas'),
        ('scikit-learn', 'sklearn'),
        ('Matplotlib', 'matplotlib'),
    ]
    
    all_installed = True
    for package_name, import_name in required_packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print()
    print("=" * 60)
    
    if all_installed:
        print("✓ All required packages are installed!")
        print()
        
        # Additional version checks
        import tensorflow as tf
        import numpy as np
        
        print("Additional Information:")
        print(f"  - TensorFlow GPU available: {tf.test.is_gpu_available()}")
        print(f"  - Python version: {sys.version.split()[0]}")
        print(f"  - NumPy BLAS info: {np.__config__.show() if hasattr(np.__config__, 'show') else 'N/A'}")
        print()
        print("✓ Installation successful! You're ready to run the pipeline.")
        return 0
    else:
        print("✗ Some packages are missing. Please install them:")
        print("  pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
