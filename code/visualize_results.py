#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate publication-quality visualizations for thesis results.

Usage:
    python visualize_results.py --output results/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

# Thesis results data
UNIMODAL_RESULTS = {
    'Clinical': {'accuracy': 0.813, 'precision': 0.712, 'sensitivity': 0.413, 'auc': 0.834},
    'CNA': {'accuracy': 0.893, 'precision': 0.841, 'sensitivity': 0.702, 'auc': 0.850},
    'Gene Expression': {'accuracy': 0.841, 'precision': 0.779, 'sensitivity': 0.505, 'auc': 0.923},
    'Multimodal': {'accuracy': 0.912, 'precision': 0.841, 'sensitivity': 0.798, 'auc': 0.950}
}

COMPARISON_METHODS = {
    'Our Method': {'accuracy': 0.912, 'precision': 0.841, 'sensitivity': 0.798, 'auc': 0.950},
    'Stacked RF': {'accuracy': 0.902, 'precision': 0.841, 'sensitivity': 0.747, 'auc': 0.930},
    'MDNNMD': {'accuracy': 0.826, 'precision': 0.749, 'sensitivity': 0.450, 'auc': 0.845},
    'SVM': {'accuracy': 0.805, 'precision': 0.708, 'sensitivity': 0.365, 'auc': 0.810},
    'Random Forest': {'accuracy': 0.791, 'precision': 0.766, 'sensitivity': 0.226, 'auc': 0.801},
    'Logistic Reg.': {'accuracy': 0.760, 'precision': 0.549, 'sensitivity': 0.183, 'auc': 0.663}
}


def plot_ablation_study(output_dir='results/figures'):
    """Plot ablation study comparing unimodal vs multimodal"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Ablation Study: Modality Contribution Analysis', fontsize=16, fontweight='bold')
    
    modalities = list(UNIMODAL_RESULTS.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    metrics = ['accuracy', 'precision', 'sensitivity', 'auc']
    metric_names = ['Accuracy', 'Precision', 'Sensitivity', 'AUC']
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        values = [UNIMODAL_RESULTS[mod][metric] for mod in modalities]
        
        bars = ax.bar(modalities, values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Highlight the multimodal bar
        bars[-1].set_edgecolor('black')
        bars[-1].set_linewidth(3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(modalities, rotation=45, ha='right')
        
        # Add improvement annotation
        if metric_name != 'Precision':  # Precision is tied
            improvement = values[-1] - max(values[:-1])
            ax.annotate(f'+{improvement:.3f}', 
                       xy=(3, values[-1]), 
                       xytext=(3.2, values[-1] - 0.05),
                       fontsize=10, color='green', fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{output_dir}/ablation_study.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/ablation_study.png")
    plt.close()


def plot_method_comparison(output_dir='results/figures'):
    """Plot comparison with state-of-the-art methods"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = list(COMPARISON_METHODS.keys())
    metrics = ['accuracy', 'precision', 'sensitivity', 'auc']
    metric_labels = ['Accuracy', 'Precision', 'Sensitivity', 'AUC']
    
    x = np.arange(len(methods))
    width = 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [COMPARISON_METHODS[method][metric] for method in methods]
        offset = width * (idx - 1.5)
        bars = ax.bar(x + offset, values, width, label=label, color=color, 
                     edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Performance Comparison: Our Method vs State-of-the-Art', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 1.0])
    
    # Highlight "Our Method"
    ax.axvspan(-0.5, 0.5, alpha=0.1, color='gold', zorder=-1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/method_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/method_comparison.png")
    plt.close()


def plot_metric_heatmap(output_dir='results/figures'):
    """Create heatmap showing all metrics for all methods"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(COMPARISON_METHODS.keys())
    metrics = ['Accuracy', 'Precision', 'Sensitivity', 'AUC']
    
    data = np.array([[COMPARISON_METHODS[method][m.lower()] for m in metrics] 
                     for method in methods])
    
    im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_yticklabels(methods, fontsize=12)
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f'{data[i, j]:.3f}',
                          ha="center", va="center", color="black", 
                          fontweight='bold', fontsize=10)
    
    ax.set_title('Performance Heatmap: All Methods × All Metrics', 
                fontsize=14, fontweight='bold', pad=15)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Performance Score', rotation=270, labelpad=20, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/performance_heatmap.png")
    plt.close()


def plot_sensitivity_comparison(output_dir='results/figures'):
    """Highlight sensitivity improvements (most dramatic improvement)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(COMPARISON_METHODS.keys())
    sensitivities = [COMPARISON_METHODS[method]['sensitivity'] for method in methods]
    
    colors = ['#f39c12' if method == 'Our Method' else '#95a5a6' for method in methods]
    bars = ax.barh(methods, sensitivities, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, sensitivities):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2., 
               f'{val:.1%}',
               ha='left', va='center', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Sensitivity (True Positive Rate)', fontsize=13, fontweight='bold')
    ax.set_title('Sensitivity Comparison: Ability to Identify High-Risk Patients', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim([0, 1.0])
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add annotation for improvement
    our_sens = COMPARISON_METHODS['Our Method']['sensitivity']
    best_other = max([COMPARISON_METHODS[m]['sensitivity'] for m in methods if m != 'Our Method'])
    improvement = our_sens - best_other
    
    ax.annotate(f'+{improvement:.1%} vs 2nd best', 
               xy=(our_sens, 0), xytext=(our_sens - 0.15, 0.5),
               fontsize=12, color='green', fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sensitivity_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/sensitivity_comparison.png")
    plt.close()


def plot_improvement_over_baselines(output_dir='results/figures'):
    """Show absolute improvement over baseline methods"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    our_results = COMPARISON_METHODS['Our Method']
    baselines = ['Logistic Reg.', 'Random Forest', 'SVM', 'MDNNMD', 'Stacked RF']
    
    metrics = ['accuracy', 'sensitivity', 'auc']
    metric_labels = ['Accuracy', 'Sensitivity', 'AUC']
    colors = ['#3498db', '#2ecc71', '#f39c12']
    
    x = np.arange(len(baselines))
    width = 0.25
    
    for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        improvements = [our_results[metric] - COMPARISON_METHODS[method][metric] 
                       for method in baselines]
        offset = width * (idx - 1)
        ax.bar(x + offset, improvements, width, label=label, color=color,
              edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Baseline Method', fontsize=13, fontweight='bold')
    ax.set_ylabel('Absolute Improvement', fontsize=13, fontweight='bold')
    ax.set_title('Performance Gain Over Baseline Methods', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(baselines, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=11, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/improvement_over_baselines.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/improvement_over_baselines.png")
    plt.close()


def main():
    """Generate all publication figures"""
    output_dir = 'results/figures'
    
    print("=" * 60)
    print("Generating Publication-Quality Figures")
    print("=" * 60)
    print()
    
    plot_ablation_study(output_dir)
    plot_method_comparison(output_dir)
    plot_metric_heatmap(output_dir)
    plot_sensitivity_comparison(output_dir)
    plot_improvement_over_baselines(output_dir)
    
    print()
    print("=" * 60)
    print(f"✓ All figures saved to: {output_dir}/")
    print("=" * 60)
    print()
    print("Generated files:")
    print("  - ablation_study.png")
    print("  - method_comparison.png")
    print("  - performance_heatmap.png")
    print("  - sensitivity_comparison.png")
    print("  - improvement_over_baselines.png")


if __name__ == "__main__":
    main()
