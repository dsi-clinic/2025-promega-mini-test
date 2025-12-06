#!/usr/bin/env python3
"""
Generate comprehensive summary table for improved EfficientNet results.
Includes all metrics: Accuracy, F1, Recall, Precision, TNR, ROC-AUC, PR-AUC, Balanced Accuracy.
"""

import json
import csv
from pathlib import Path
import argparse
from collections import defaultdict

def count_samples_by_day(split_file):
    """Count samples and organoids per day from split file."""
    with open(split_file) as f:
        split_data = json.load(f)
    
    day_counts = defaultdict(lambda: {"samples": 0, "organoids": set()})
    
    for organoid_id, organoid_data in split_data.items():
        for day, day_data in organoid_data.get("timepoints", {}).items():
            day_counts[day]["samples"] += 1
            day_counts[day]["organoids"].add(organoid_id)
    
    result = {}
    for day, counts in day_counts.items():
        result[day] = {
            "samples": counts["samples"],
            "organoids": len(counts["organoids"])
        }
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Directory with training results")
    parser.add_argument("--split-prefix", required=True, help="Prefix for split files (e.g., 'both_train_base_no_stitch')")
    parser.add_argument("--output-name", required=True, help="Output CSV filename (without .csv)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    split_dir = Path("data_splits")
    
    # Load sample counts
    train_file = split_dir / f"{args.split_prefix.replace('_train_', '_train_')}.json"
    val_file = split_dir / f"{args.split_prefix.replace('_train_', '_val_')}.json"
    test_file = split_dir / f"{args.split_prefix.replace('_train_', '_test_')}.json"
    
    train_counts = count_samples_by_day(train_file)
    val_counts = count_samples_by_day(val_file)
    test_counts = count_samples_by_day(test_file)
    
    # Load results
    efficientnet_dir = results_dir / "efficientnet"
    if not efficientnet_dir.exists():
        print(f"ERROR: Results directory not found: {efficientnet_dir}")
        return
    
    summary_rows = []
    
    for day_dir in sorted(efficientnet_dir.iterdir()):
        if not day_dir.is_dir():
            continue
        
        day_str = day_dir.name
        metrics_file = day_dir / "metrics_test.json"
        
        if not metrics_file.exists():
            print(f"WARNING: No metrics file for {day_str}")
            continue
        
        with open(metrics_file) as f:
            metrics = json.load(f)
        
        # Calculate confusion matrix from metrics
        test_n = metrics.get('test_n', 0)
        actual_good = metrics.get('actual_good', 0)
        predicted_good = metrics.get('predicted_good', 0)
        tpr = metrics.get('tpr', 0)
        tnr = metrics.get('tnr', 0)
        
        # Calculate TP, FP, TN, FN
        tp = int(round(tpr * actual_good)) if actual_good > 0 else 0
        fn = actual_good - tp
        fp = predicted_good - tp
        tn = test_n - actual_good - fp
        
        # Calculate precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Get counts
        train_samples = train_counts.get(day_str, {}).get("samples", 0)
        train_organoids = train_counts.get(day_str, {}).get("organoids", 0)
        val_samples = val_counts.get(day_str, {}).get("samples", 0)
        val_organoids = val_counts.get(day_str, {}).get("organoids", 0)
        test_samples = test_counts.get(day_str, {}).get("samples", 0)
        test_organoids = test_counts.get(day_str, {}).get("organoids", 0)
        
        row = {
            "Day": day_str,
            "Day_No": metrics.get("day_no", ""),
            "Backbone": "efficientnet",
            # Test metrics
            "Test_Accuracy": f"{metrics.get('accuracy', 0):.4f}",
            "Test_F1": f"{metrics.get('f1', 0):.4f}",
            "Test_Recall": f"{metrics.get('tpr', 0):.4f}",  # TPR = Recall
            "Test_Precision": f"{precision:.4f}",
            "Test_TNR": f"{metrics.get('tnr', 0):.4f}",  # NEW: TNR
            "Test_Balanced_Accuracy": f"{metrics.get('balanced_accuracy', 0):.4f}",  # NEW
            "Test_ROC_AUC": f"{metrics.get('roc_auc', 0):.4f}" if metrics.get('roc_auc') else "N/A",
            "Test_PR_AUC": f"{metrics.get('pr_auc', 0):.4f}" if metrics.get('pr_auc') else "N/A",
            # Confusion matrix
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            # Sample counts
            "Train_Samples": train_samples,
            "Val_Samples": val_samples,
            "Test_Samples": test_samples,
            # Organoid counts
            "Train_Organoids": train_organoids,
            "Val_Organoids": val_organoids,
            "Test_Organoids": test_organoids,
            # Additional info
            "Test_N": metrics.get('test_n', test_samples),
            "Actual_Good": metrics.get('actual_good', 0),
            "Predicted_Good": metrics.get('predicted_good', 0),
            "Val_Balanced_Accuracy_For_Selection": f"{metrics.get('val_balanced_accuracy_for_selection', 0):.4f}",
        }
        
        summary_rows.append(row)
    
    if not summary_rows:
        print("ERROR: No results found!")
        return
    
    # Sort by day number
    def get_day_num(row):
        day_no = row['Day_No']
        if day_no == "":
            return 999
        return float(day_no)
    
    summary_rows = sorted(summary_rows, key=get_day_num)
    
    # Write CSV
    output_file = Path.home() / f"{args.output_name}.csv"
    fieldnames = list(summary_rows[0].keys())
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"\n[OK] Summary table saved to: {output_file}")
    print(f"  Total rows: {len(summary_rows)}")
    print(f"  Days covered: {len(summary_rows)}")
    
    # Print summary statistics
    if summary_rows:
        avg_tnr = sum(float(r['Test_TNR']) for r in summary_rows) / len(summary_rows)
        avg_tpr = sum(float(r['Test_Recall']) for r in summary_rows) / len(summary_rows)
        avg_balanced = sum(float(r['Test_Balanced_Accuracy']) for r in summary_rows) / len(summary_rows)
        
        print(f"\n  Average Test TNR: {avg_tnr:.4f}")
        print(f"  Average Test TPR: {avg_tpr:.4f}")
        print(f"  Average Balanced Accuracy: {avg_balanced:.4f}")

if __name__ == "__main__":
    main()

