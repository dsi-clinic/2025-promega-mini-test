"""
Generate publication-quality comparison plots from CNN-LSTM results
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set publication-quality defaults
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 300

def load_results(path):
    """Load results JSON"""
    with open(path) as f:
        return json.load(f)

def plot_training_curves(baseline_results, improved_results, output_dir):
    """
    Plot training and validation curves for both models
    Blue = Training, Orange = Validation
    """
    # Extract histories
    hist_base = baseline_results['train_history']
    hist_imp = improved_results['train_history']
    
    epochs_base = [h['epoch'] for h in hist_base]
    epochs_imp = [h['epoch'] for h in hist_imp]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Colors
    color_train = '#3498db'  # Blue
    color_val = '#e67e22'    # Orange
    
    # ========== BASELINE LOSS ==========
    ax = axes[0, 0]
    ax.plot(epochs_base, [h['train_loss'] for h in hist_base], 
            color=color_train, linewidth=2.5, label='Training')
    ax.plot(epochs_base, [h['val_loss'] for h in hist_base], 
            color=color_val, linewidth=2.5, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Baseline (White Padding): Loss', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2.5)
    
    # ========== BASELINE ACCURACY ==========
    ax = axes[0, 1]
    ax.plot(epochs_base, [h['train_acc'] for h in hist_base], 
            color=color_train, linewidth=2.5, label='Training')
    ax.plot(epochs_base, [h['val_acc'] for h in hist_base], 
            color=color_val, linewidth=2.5, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Baseline (White Padding): Accuracy', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # ========== IMPROVED LOSS ==========
    ax = axes[1, 0]
    ax.plot(epochs_imp, [h['train_loss'] for h in hist_imp], 
            color=color_train, linewidth=2.5, label='Training')
    ax.plot(epochs_imp, [h['val_loss'] for h in hist_imp], 
            color=color_val, linewidth=2.5, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Improved (Smoothed Edges): Loss', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 2.5)
    
    # ========== IMPROVED ACCURACY ==========
    ax = axes[1, 1]
    ax.plot(epochs_imp, [h['train_acc'] for h in hist_imp], 
            color=color_train, linewidth=2.5, label='Training')
    ax.plot(epochs_imp, [h['val_acc'] for h in hist_imp], 
            color=color_val, linewidth=2.5, label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Improved (Smoothed Edges): Accuracy', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / 'training_curves_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_confusion_matrices(baseline_results, improved_results, output_dir):
    """Plot confusion matrices side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    cm_base = np.array(baseline_results['confusion_matrix'])
    cm_imp = np.array(improved_results['confusion_matrix'])
    
    # Baseline
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'],
                ax=axes[0], cbar_kws={'label': 'Count'}, 
                annot_kws={'size': 16, 'weight': 'bold'},
                vmin=0, vmax=max(cm_base.max(), cm_imp.max()))
    axes[0].set_xlabel('Predicted', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Actual', fontsize=14, fontweight='bold')
    axes[0].set_title(f'Baseline (White Padding)\nTest Accuracy: {baseline_results["test_acc"]:.1%}', 
                      fontsize=16, fontweight='bold', pad=15)
    
    # Improved
    sns.heatmap(cm_imp, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'],
                ax=axes[1], cbar_kws={'label': 'Count'},
                annot_kws={'size': 16, 'weight': 'bold'},
                vmin=0, vmax=max(cm_base.max(), cm_imp.max()))
    axes[1].set_xlabel('Predicted', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=14, fontweight='bold')
    axes[1].set_title(f'Improved (Smoothed Edges)\nTest Accuracy: {improved_results["test_acc"]:.1%}', 
                      fontsize=16, fontweight='bold', pad=15)
    
    plt.tight_layout()
    output_path = output_dir / 'confusion_matrices_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def plot_metrics_table(baseline_results, improved_results, output_dir):
    """Create a metrics comparison table as an image"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    # Data for table
    metrics = [
        ('Test Accuracy', baseline_results['test_acc'], improved_results['test_acc']),
        ('Test Precision', baseline_results['test_precision'], improved_results['test_precision']),
        ('Test Recall', baseline_results['test_recall'], improved_results['test_recall']),
        ('Test F1 Score', baseline_results['test_f1'], improved_results['test_f1']),
        ('Best Val Accuracy', baseline_results['best_val_acc'], improved_results['best_val_acc']),
    ]
    
    table_data = []
    for name, base_val, imp_val in metrics:
        delta = imp_val - base_val
        delta_str = f"{delta:+.4f}"
        arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
        table_data.append([
            name,
            f"{base_val:.4f}",
            f"{imp_val:.4f}",
            f"{delta_str} {arrow}"
        ])
    
    # Add error counts
    cm_base = np.array(baseline_results['confusion_matrix'])
    cm_imp = np.array(improved_results['confusion_matrix'])
    
    fn_base, fn_imp = cm_base[0, 1], cm_imp[0, 1]
    fp_base, fp_imp = cm_base[1, 0], cm_imp[1, 0]
    total_base = fn_base + fp_base
    total_imp = fn_imp + fp_imp
    
    table_data.append(['False Negatives', str(fn_base), str(fn_imp), f"{fn_imp-fn_base:+d}"])
    table_data.append(['False Positives', str(fp_base), str(fp_imp), f"{fp_imp-fp_base:+d}"])
    table_data.append(['Total Errors', str(total_base), str(total_imp), f"{total_imp-total_base:+d}"])
    
    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Metric', 'Baseline\n(White)', 'Improved\n(Blur)', 'Change'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(4):
        cell = table[(0, i)]
        cell.set_facecolor('#3498db')
        cell.set_text_props(weight='bold', color='white', size=12)
    
    # Color improvement rows
    for i in range(1, len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
    
    ax.set_title('Model Performance Comparison', fontsize=18, fontweight='bold', pad=20)
    
    output_path = output_dir / 'metrics_comparison_table.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def print_summary(baseline_results, improved_results):
    """Print summary to console"""
    print("\n" + "="*70)
    print("MODEL COMPARISON SUMMARY")
    print("="*70)
    print(f"Baseline Test Accuracy:  {baseline_results['test_acc']:.4f} ({baseline_results['test_acc']*100:.2f}%)")
    print(f"Improved Test Accuracy:  {improved_results['test_acc']:.4f} ({improved_results['test_acc']*100:.2f}%)")
    print(f"Improvement:             {(improved_results['test_acc'] - baseline_results['test_acc']):.4f} ({(improved_results['test_acc'] - baseline_results['test_acc'])*100:.2f} percentage points)")
    print("="*70)
    
    cm_base = np.array(baseline_results['confusion_matrix'])
    cm_imp = np.array(improved_results['confusion_matrix'])
    
    print("\nConfusion Matrix Changes:")
    print(f"  False Negatives: {cm_base[0,1]} → {cm_imp[0,1]} ({cm_imp[0,1]-cm_base[0,1]:+d})")
    print(f"  False Positives: {cm_base[1,0]} → {cm_imp[1,0]} ({cm_imp[1,0]-cm_base[1,0]:+d})")
    print(f"  Total Errors:    {cm_base[0,1]+cm_base[1,0]} → {cm_imp[0,1]+cm_imp[1,0]} ({(cm_imp[0,1]+cm_imp[1,0])-(cm_base[0,1]+cm_base[1,0]):+d})")
    print("="*70 + "\n")

def main():
    # Paths
    base_dir = Path('/net/projects2/promega/data-analysis/output/cnn_lstm/models')
    baseline_path = base_dir / 'results_allwhite.json'
    improved_path = base_dir / 'results_blur.json'
    
    output_dir = base_dir / 'comparison_plots'
    output_dir.mkdir(exist_ok=True)
    
    # Check files exist
    if not baseline_path.exists():
        print(f"ERROR: Baseline results not found at {baseline_path}")
        return
    
    if not improved_path.exists():
        print(f"ERROR: Improved results not found at {improved_path}")
        return
    
    # Load results
    print("Loading results...")
    baseline_results = load_results(baseline_path)
    improved_results = load_results(improved_path)
    
    # Print summary
    print_summary(baseline_results, improved_results)
    
    # Generate plots
    print("Generating plots...")
    plot_training_curves(baseline_results, improved_results, output_dir)
    plot_confusion_matrices(baseline_results, improved_results, output_dir)
    plot_metrics_table(baseline_results, improved_results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - training_curves_comparison.png")
    print(f"  - confusion_matrices_comparison.png")
    print(f"  - metrics_comparison_table.png")

if __name__ == '__main__':
    main()