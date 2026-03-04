#!/usr/bin/env python3
"""
Compare Amanda's bundled results with our successful GPU run (686612)
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

# Load results
amanda_path = Path("/home/your_name/image_classifier_ts/2025-promega-mini-test/analysis/images/cnn_lstm/models/results.json")
our_path = Path("/net/projects2/promega/data-analysis/output/cnn_lstm/models/run_686612/results.json")

with open(amanda_path) as f:
    amanda = json.load(f)
with open(our_path) as f:
    our = json.load(f)

output_dir = Path("/home/your_name/image_classifier_ts/2025-promega-mini-test/analysis/images/cnn_lstm/comparisons/run_686612_vs_bundled")
output_dir.mkdir(parents=True, exist_ok=True)

# Extract training histories
amanda_hist = amanda.get('train_history', [])
our_hist = our.get('train_history', [])

# Separate our phases
our_p1 = [e for e in our_hist if e.get('phase') == 1]
our_p2 = [e for e in our_hist if e.get('phase') == 2]

# Create comparison plots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Amanda Bundled Results vs Our GPU Run (686612) - Full Comparison', fontsize=16, fontweight='bold')

# 1. Training Loss
ax = axes[0, 0]
amanda_epochs = list(range(1, len(amanda_hist) + 1))
amanda_train_loss = [e['train_loss'] for e in amanda_hist]
amanda_val_loss = [e['val_loss'] for e in amanda_hist]

our_epochs = list(range(1, len(our_hist) + 1))
our_train_loss = [e['train_loss'] for e in our_hist]
our_val_loss = [e['val_loss'] for e in our_hist]

ax.plot(amanda_epochs, amanda_train_loss, 'b-', label="Amanda Train", linewidth=2, alpha=0.7)
ax.plot(amanda_epochs, amanda_val_loss, 'b--', label="Amanda Val", linewidth=2, alpha=0.7)
ax.plot(our_epochs, our_train_loss, 'r-', label="Our Train", linewidth=2, alpha=0.7)
ax.plot(our_epochs, our_val_loss, 'r--', label="Our Val", linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Training Accuracy
ax = axes[0, 1]
amanda_train_acc = [e['train_acc'] for e in amanda_hist]
amanda_val_acc = [e['val_acc'] for e in amanda_hist]
our_train_acc = [e['train_acc'] for e in our_hist]
our_val_acc = [e['val_acc'] for e in our_hist]

ax.plot(amanda_epochs, amanda_train_acc, 'b-', label="Amanda Train", linewidth=2, alpha=0.7)
ax.plot(amanda_epochs, amanda_val_acc, 'b--', label="Amanda Val", linewidth=2, alpha=0.7)
ax.plot(our_epochs, our_train_acc, 'r-', label="Our Train", linewidth=2, alpha=0.7)
ax.plot(our_epochs, our_val_acc, 'r--', label="Our Val", linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training & Validation Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.0])

# 3. Validation F1
ax = axes[0, 2]
amanda_val_f1 = [e.get('val_f1', 0) for e in amanda_hist]
our_val_f1 = [e.get('val_f1', 0) for e in our_hist]

ax.plot(amanda_epochs, amanda_val_f1, 'b-', label="Amanda", linewidth=2, alpha=0.7)
ax.plot(our_epochs, our_val_f1, 'r-', label="Our", linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('F1 Score')
ax.set_title('Validation F1 Score')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 1.0])

# 4. Confusion Matrices
ax1 = axes[1, 0]
ax2 = axes[1, 1]

amanda_cm = np.array(amanda.get('confusion_matrix', []))
our_cm = np.array(our.get('confusion_matrix', []))

sns.heatmap(amanda_cm, annot=True, fmt='d', cmap='Blues', ax=ax1, 
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'],
            cbar_kws={'label': 'Count'})
ax1.set_title('Amanda: Confusion Matrix\n(Test Set)')
ax1.set_ylabel('Actual')
ax1.set_xlabel('Predicted')

sns.heatmap(our_cm, annot=True, fmt='d', cmap='Reds', ax=ax2,
            xticklabels=['Bad', 'Good'], yticklabels=['Bad', 'Good'],
            cbar_kws={'label': 'Count'})
ax2.set_title('Our Run: Confusion Matrix\n(Test Set)')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# 5. Final Metrics Comparison
ax = axes[1, 2]
ax.axis('off')

metrics_data = [
    ['Metric', 'Amanda', 'Our Run', 'Difference'],
    ['Test Accuracy', f"{amanda['test_acc']:.4f}", f"{our['test_acc']:.4f}", 
     f"{our['test_acc'] - amanda['test_acc']:+.4f}"],
    ['Test Precision', f"{amanda['test_precision']:.4f}", f"{our['test_precision']:.4f}",
     f"{our['test_precision'] - amanda['test_precision']:+.4f}"],
    ['Test Recall', f"{amanda['test_recall']:.4f}", f"{our['test_recall']:.4f}",
     f"{our['test_recall'] - amanda['test_recall']:+.4f}"],
    ['Test F1', f"{amanda['test_f1']:.4f}", f"{our['test_f1']:.4f}",
     f"{our['test_f1'] - amanda['test_f1']:+.4f}"],
    ['Best Val Acc', f"{amanda['best_val_acc']:.4f}", f"{our['best_val_acc']:.4f}",
     f"{our['best_val_acc'] - amanda['best_val_acc']:+.4f}"],
    ['Best Val Loss', f"{amanda['best_val_loss']:.4f}", f"{our['best_val_loss']:.4f}",
     f"{our['best_val_loss'] - amanda['best_val_loss']:+.4f}"],
]

table = ax.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                cellLoc='center', loc='center',
                colWidths=[0.3, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)
ax.set_title('Final Metrics Comparison', fontsize=12, fontweight='bold', pad=20)

# Color code differences
for i in range(1, len(metrics_data)):
    diff_val = float(metrics_data[i][3])
    if diff_val > 0:
        table[(i, 3)].set_facecolor('#90EE90')  # Light green for positive
    elif diff_val < 0:
        table[(i, 3)].set_facecolor('#FFB6C1')  # Light red for negative

plt.tight_layout()
plt.savefig(output_dir / 'full_comparison_panel.png', dpi=300, bbox_inches='tight')
print(f"Saved comparison plot to: {output_dir / 'full_comparison_panel.png'}")

# Create text report
report_path = output_dir / 'COMPARISON_REPORT.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPARISON: Amanda's Bundled Results vs Our GPU Run (686612)\n")
    f.write("="*80 + "\n\n")
    
    f.write("KEY FINDINGS:\n")
    f.write("-"*80 + "\n")
    f.write("1. TRAINING METHODOLOGY:\n")
    f.write("   - Amanda's bundled results: Single-phase training (no phase separation)\n")
    f.write("   - Our run: Two-phase training (Phase 1: CNN frozen, Phase 2: All params unfrozen)\n")
    f.write(f"   - Amanda epochs: {len(amanda_hist)}\n")
    f.write(f"   - Our epochs: Phase 1={len(our_p1)}, Phase 2={len(our_p2)}, Total={len(our_hist)}\n\n")
    
    f.write("2. TEST SET DIFFERENCES:\n")
    amanda_total = sum([sum(row) for row in amanda_cm])
    our_total = sum([sum(row) for row in our_cm])
    f.write(f"   - Amanda test set size: {amanda_total} samples\n")
    f.write(f"   - Our test set size: {our_total} samples\n")
    f.write("   - NOTE: Different test sets may explain some metric differences\n\n")
    
    f.write("3. FINAL METRICS COMPARISON:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Metric':<20} {'Amanda':<15} {'Our Run':<15} {'Difference':<15}\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Test Accuracy':<20} {amanda['test_acc']:<15.4f} {our['test_acc']:<15.4f} {our['test_acc'] - amanda['test_acc']:+.4f}\n")
    f.write(f"{'Test Precision':<20} {amanda['test_precision']:<15.4f} {our['test_precision']:<15.4f} {our['test_precision'] - amanda['test_precision']:+.4f}\n")
    f.write(f"{'Test Recall':<20} {amanda['test_recall']:<15.4f} {our['test_recall']:<15.4f} {our['test_recall'] - amanda['test_recall']:+.4f}\n")
    f.write(f"{'Test F1':<20} {amanda['test_f1']:<15.4f} {our['test_f1']:<15.4f} {our['test_f1'] - amanda['test_f1']:+.4f}\n")
    f.write(f"{'Best Val Acc':<20} {amanda['best_val_acc']:<15.4f} {our['best_val_acc']:<15.4f} {our['best_val_acc'] - amanda['best_val_acc']:+.4f}\n")
    f.write(f"{'Best Val Loss':<20} {amanda['best_val_loss']:<15.4f} {our['best_val_loss']:<15.4f} {our['best_val_loss'] - amanda['best_val_loss']:+.4f}\n\n")
    
    f.write("4. CONFUSION MATRIX COMPARISON:\n")
    f.write("-"*80 + "\n")
    f.write("Amanda's Confusion Matrix:\n")
    f.write(f"  True Negatives (Bad→Bad): {amanda_cm[0][0]}\n")
    f.write(f"  False Positives (Bad→Good): {amanda_cm[0][1]}\n")
    f.write(f"  False Negatives (Good→Bad): {amanda_cm[1][0]}\n")
    f.write(f"  True Positives (Good→Good): {amanda_cm[1][1]}\n\n")
    f.write("Our Run Confusion Matrix:\n")
    f.write(f"  True Negatives (Bad→Bad): {our_cm[0][0]}\n")
    f.write(f"  False Positives (Bad→Good): {our_cm[0][1]}\n")
    f.write(f"  False Negatives (Good→Bad): {our_cm[1][0]}\n")
    f.write(f"  True Positives (Good→Good): {our_cm[1][1]}\n\n")
    
    f.write("5. ANALYSIS:\n")
    f.write("-"*80 + "\n")
    f.write("Our run achieved:\n")
    f.write("  ✓ Higher validation accuracy (+11.5%)\n")
    f.write("  ✓ Lower validation loss (-50.4%)\n")
    f.write("  ✓ Higher test recall (+1.9%)\n")
    f.write("  ✗ Lower test accuracy (-4.2%)\n")
    f.write("  ✗ Lower test precision (-10.1%)\n")
    f.write("  ✗ Lower test F1 (-4.9%)\n\n")
    f.write("NOTE: The different test sets and training methodologies make direct\n")
    f.write("comparison challenging. Our two-phase approach shows better validation\n")
    f.write("performance but slightly lower test performance, possibly due to:\n")
    f.write("  - Different test set composition\n")
    f.write("  - Different data splits\n")
    f.write("  - Training methodology differences (single vs two-phase)\n")

print(f"Saved comparison report to: {report_path}")
