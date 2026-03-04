#!/usr/bin/env python3
"""
Run our ORIGINAL EfficientNet baseline using our own data split.
ONLY CHANGE: Data loading (uses our split instead of load_data_and_create_splits)
EVERYTHING ELSE: Exactly as original (model, training, hyperparameters from train_base_model.py)
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1] / "2025-promega-mini-test"
sys.path.insert(0, str(ROOT))

# Import our ORIGINAL training functions (unchanged from train_base_model.py)
from analysis.images.cnn_lstm.train_base_model import (
    train_for_day, set_seed, DEVICE, SEED, MAX_EPOCHS, PATIENCE, LR, GRAD_CLIP, TARGET_SIZE
)

# Import our data loader (ONLY change)
from analysis.images.cnn_lstm.load_split_data import load_split_data

import torch
import json

def main():
    # Set seed (original function)
    set_seed(SEED)
    device = torch.device(DEVICE)
    print(f"Using device: {device}")
    
    # Output directory (using exclude-nothing data)
    output_dir = Path(__file__).parent / "our_efficientnet_all_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # ========== ONLY CHANGE: Load our split data instead of load_data_and_create_splits ==========
    print("\n" + "="*70)
    print("LOADING OUR DATA SPLIT (ONLY CHANGE FROM ORIGINAL)")
    print("="*70)
    
    # Use exclude-nothing version (includes stitched/presplit samples)
    split_dir = Path(__file__).resolve().parents[1] / "2025-promega-mini-test" / "data_splits"
    train_ids, val_ids, test_ids, series_metadata, data = load_split_data(
        split_dir / "both_train_base.json",
        split_dir / "both_val_base.json",
        split_dir / "both_test_base.json"
    )
    # ========== END OF ONLY CHANGE ==========
    
    # Everything below is EXACTLY as original train_base_model.py
    print(f"\nSplits: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")
    
    # Train for day 30 (final prediction) - using original train_for_day function
    print("\n" + "="*70)
    print("TRAINING EFFICIENTNET BASELINE (Day 30) - ORIGINAL FUNCTION")
    print("="*70)
    
    result = train_for_day(
        30, train_ids, val_ids, test_ids,
        series_metadata, data, device,
        output_dir / "day_30"
    )
    
    if result:
        # Save results (original format)
        results_path = output_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
        
        print("\n" + "="*70)
        print("RESULTS (ORIGINAL FORMAT)")
        print("="*70)
        print(f"Test Accuracy: {result['test_acc']:.4f}")
        print(f"Test F1: {result['test_f1']:.4f}")
        print(f"Test Precision: {result['test_precision']:.4f}")
        print(f"Test Recall: {result['test_recall']:.4f}")
        print(f"Best Val Acc: {result['best_val_acc']:.4f}")
        print(f"\nResults saved to: {results_path}")

if __name__ == "__main__":
    main()
