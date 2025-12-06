#!/usr/bin/env python3
"""
Analyze Metabolite Classifier Results
Aggregates results from `outputs_metabolites` and generates comparison reports.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_results():
    script_dir = Path(__file__).parent
    output_dir = script_dir / 'outputs_metabolites'
    report_dir = script_dir.parent / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"Analyzing results from: {output_dir}")
    
    all_summaries = []

    if not output_dir.exists():
        print(f"Output directory {output_dir} does not exist.")
        return

    # Filter for relevant model directories
    for model_dir in output_dir.iterdir():
        if model_dir.is_dir() and model_dir.name.startswith("lgbm_per_day_noscale_cv_"):
            summary_path = model_dir / 'results_summary.csv'
            if summary_path.exists():
                print(f"Found results for: {model_dir.name}")
                df = pd.read_csv(summary_path)
                df['Model_Variant'] = model_dir.name
                all_summaries.append(df)
            else:
                print(f"Skipping {model_dir.name} (no results_summary.csv)")

    if not all_summaries:
        print("No matching results found to analyze.")
        return

    combined_df = pd.concat(all_summaries, ignore_index=True)
    
    # Save combined raw data
    combined_csv_path = report_dir / 'combined_model_metrics.csv'
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"\nSaved combined metrics to {combined_csv_path}")

    # --- Comparison Plots ---
    
    # Metrics to plot
    metrics = [
        ('Test_F1_NotAcceptable', 'F1 Score (Not Acceptable)'),
        ('Test_F1_Acceptable', 'F1 Score (Acceptable)'),
        ('Test_Specificity', 'Specificity (TNR)'),
        ('Test_ROC_AUC', 'ROC AUC')
    ]

    for metric_col, metric_name in metrics:
        if metric_col not in combined_df.columns:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Plot each variant
        for variant in combined_df['Model_Variant'].unique():
            subset = combined_df[combined_df['Model_Variant'] == variant].sort_values('Day_No')
            plt.plot(subset['Day_No'], subset[metric_col], marker='o', label=variant, alpha=0.7)
            
        plt.title(f'{metric_name} by Day across Variants')
        plt.xlabel('Day')
        plt.ylabel(metric_name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = report_dir / f'comparison_{metric_col}.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"Saved comparison plot: {plot_path}")

    # --- Summary Table (Average across days) ---
    avg_metrics = combined_df.groupby('Model_Variant')[
        ['Test_F1_NotAcceptable', 'Test_F1_Acceptable', 'Test_Specificity', 'Test_ROC_AUC', 'Test_Accuracy']
    ].mean().reset_index()
    
    # Sort by F1 Not Acceptable by default
    avg_metrics = avg_metrics.sort_values('Test_F1_NotAcceptable', ascending=False)
    
    print("\nAverage Metrics across all days (Sorted by F1 Not Acceptable):")
    print(avg_metrics.to_string(index=False, float_format="%.3f"))
    
    avg_csv_path = report_dir / 'average_metrics_summary.csv'
    avg_metrics.to_csv(avg_csv_path, index=False)
    print(f"Saved average metrics to {avg_csv_path}")

    # --- Identify Best Models ---
    print("\nBest Models per Metric (Average across days):")
    best_models = []
    for col in ['Test_F1_NotAcceptable', 'Test_F1_Acceptable', 'Test_Specificity', 'Test_ROC_AUC', 'Test_Accuracy']:
        best_row = avg_metrics.loc[avg_metrics[col].idxmax()]
        print(f"  {col}: {best_row['Model_Variant']} ({best_row[col]:.3f})")
        best_models.append({
            'Metric': col,
            'Best_Model': best_row['Model_Variant'],
            'Score': best_row[col]
        })
    
    best_models_df = pd.DataFrame(best_models)
    best_models_csv_path = report_dir / 'best_models_summary.csv'
    best_models_df.to_csv(best_models_csv_path, index=False)
    print(f"Saved best models summary to {best_models_csv_path}")

if __name__ == "__main__":
    analyze_results()
