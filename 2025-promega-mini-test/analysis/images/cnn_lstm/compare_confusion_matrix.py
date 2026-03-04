"""
Quick script to compare confusion matrices between Amanda's results and two-phase run
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Load results
amanda_path = ROOT / "analysis/images/cnn_lstm/models/results.json"
two_phase_path = Path("/net/projects2/promega/data-analysis/output/cnn_lstm/models/run_686612/results.json")

amanda = json.loads(amanda_path.read_text())
two_phase = json.loads(two_phase_path.read_text())

amanda_cm = np.array(amanda['confusion_matrix'])
two_phase_cm = np.array(two_phase['confusion_matrix'])

# Create comparison figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Amanda's confusion matrix
ax1 = axes[0]
im1 = ax1.imshow(amanda_cm, cmap='Blues', aspect='auto', vmin=0, vmax=max(amanda_cm.max(), two_phase_cm.max()))
ax1.set_title("Amanda's Results (Single-Phase)", fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Bad', 'Good'])
ax1.set_yticklabels(['Bad', 'Good'])
ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('Actual', fontsize=12)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, amanda_cm[i, j], ha="center", va="center", color="black", fontsize=14, fontweight='bold')

# Two-phase confusion matrix
ax2 = axes[1]
im2 = ax2.imshow(two_phase_cm, cmap='Blues', aspect='auto', vmin=0, vmax=max(amanda_cm.max(), two_phase_cm.max()))
ax2.set_title("Two-Phase Training (Run 686612)", fontsize=14, fontweight='bold')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Bad', 'Good'])
ax2.set_yticklabels(['Bad', 'Good'])
ax2.set_xlabel('Predicted', fontsize=12)
ax2.set_ylabel('Actual', fontsize=12)

# Add text annotations
for i in range(2):
    for j in range(2):
        text = ax2.text(j, i, two_phase_cm[i, j], ha="center", va="center", color="black", fontsize=14, fontweight='bold')

# Add colorbar
plt.colorbar(im2, ax=axes, fraction=0.046, pad=0.04)

# Calculate totals
amanda_total = sum(sum(row) for row in amanda_cm)
two_phase_total = sum(sum(row) for row in two_phase_cm)

# Add metrics text below
fig.text(0.5, 0.02, 
         f"Amanda: Acc={amanda['test_acc']:.3f}, F1={amanda['test_f1']:.3f}, N={amanda_total} | "
         f"Two-Phase: Acc={two_phase['test_acc']:.3f}, F1={two_phase['test_f1']:.3f}, N={two_phase_total}",
         ha='center', fontsize=11)

# Add warning if test set sizes differ
if amanda_total != two_phase_total:
    fig.text(0.5, -0.05, 
             f"⚠️  WARNING: Different test set sizes ({amanda_total} vs {two_phase_total}) - Different split ratios used!",
             ha='center', fontsize=10, color='red', weight='bold')
    fig.text(0.5, -0.10,
             "Amanda likely used ~0.66/0.17/0.17 ratios vs current 0.7/0.15/0.15 - Results not directly comparable",
             ha='center', fontsize=9, color='red', style='italic')

plt.tight_layout(rect=[0, 0.05, 1, 1])

# Save
outdir = ROOT / "analysis/images/cnn_lstm/comparisons/amanda_vs_two_phase"
outdir.mkdir(parents=True, exist_ok=True)
outfile = outdir / "confusion_matrix_comparison.png"
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"✅ Saved comparison to: {outfile}")

plt.close()
