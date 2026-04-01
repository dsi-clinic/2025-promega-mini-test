#!/usr/bin/env python3
"""
Multimodal organoid quality classification — training script.

Orchestrates:
  1. Shared backbone pretraining across all days combined
  2. Per-day classifier head training (backbone frozen)

All model/data components are imported from multimodal_model.py.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

from multimodal_model import (
    SEED,
    set_seed,
    day_to_int,
    OrganoidDataset,
    MultimodalClassifier,
    EarlyStopping,
    get_transforms,
    load_and_prepare_data,
    train_epoch,
    eval_epoch,
    eval_epoch_detailed,
)


# ---------------------------------------------------------------------------
# Shared backbone pretraining
# ---------------------------------------------------------------------------

def pretrain_shared_backbone(train_df, val_df, config):
    """Pretrain on all days combined; return (best_state_dict, metabolite_scaler)."""
    print(f"\n{'='*60}\nPretraining Shared Backbone (All Days)\n{'='*60}")

    t_train = get_transforms(config, augment=True)  if config['use_images'] else None
    t_eval  = get_transforms(config, augment=False) if config['use_images'] else None

    train_ds = OrganoidDataset(train_df, config, t_train, fit_scaler=True)
    val_ds   = OrganoidDataset(val_df,   config, t_eval,  scaler=train_ds.scaler)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=4)

    labels_arr = [train_ds.label_map.get(train_ds.df.iloc[i]['label'], 0) for i in range(len(train_ds))]
    weights_arr = compute_class_weight('balanced', classes=np.unique(labels_arr), y=labels_arr)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(labels_arr), weights_arr)}

    model = MultimodalClassifier(config).to(config['device'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config['early_stopping_patience'])

    for epoch in range(config['num_epochs_phase1']):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, class_weights, config)
        vr = eval_epoch(model, val_loader, config)

        if vr['acc'] > best_val_acc:
            best_val_acc, best_state = vr['acc'], model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"  Ep {epoch:3d}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")

        if es(vr['acc']):
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"Pretraining done. Best val acc: {best_val_acc:.3f}\n")
    return best_state, train_ds.scaler


# ---------------------------------------------------------------------------
# Per-day training
# ---------------------------------------------------------------------------

def train_for_day(day, train_df, val_df, test_df, config, output_dir,
                  shared_backbone_state=None, shared_scaler=None):
    """Train and evaluate a day-specific classifier head; save results to output_dir/day/."""
    print(f"\n{'='*60}\nTraining for {day}\n{'='*60}")

    train_day = train_df[train_df['day'] == day]
    val_day   = val_df[val_df['day']   == day]
    test_day  = test_df[test_df['day'] == day]

    if len(train_day) == 0:
        print(f"No training data for {day}, skipping.")
        return None

    print(f"  Label counts — Train: {train_day['label'].value_counts().to_dict()}"
          f"  Val: {val_day['label'].value_counts().to_dict()}"
          f"  Test: {test_day['label'].value_counts().to_dict()}")

    t_train = get_transforms(config, augment=True)  if config['use_images'] else None
    t_eval  = get_transforms(config, augment=False) if config['use_images'] else None

    if shared_scaler is not None:
        train_ds = OrganoidDataset(train_day, config, t_train, scaler=shared_scaler)
        scaler = shared_scaler
    else:
        train_ds = OrganoidDataset(train_day, config, t_train, fit_scaler=True)
        scaler = train_ds.scaler

    val_ds  = OrganoidDataset(val_day,  config, t_eval, scaler=scaler)
    test_ds = OrganoidDataset(test_day, config, t_eval, scaler=scaler)
    print(f"  Samples — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # Class weights from training split
    labels_arr = [train_ds.label_map.get(train_day.iloc[i]['label'], 0) for i in range(len(train_day))]
    weights_arr = compute_class_weight('balanced', classes=np.unique(labels_arr), y=labels_arr)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(labels_arr), weights_arr)}

    # Build model and optionally load shared backbone weights
    model = MultimodalClassifier(config).to(config['device'])
    if shared_backbone_state is not None:
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in shared_backbone_state.items()
                      if k in model_dict and (k.startswith('backbone.') or k.startswith('meta_branch.'))}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict, strict=False)
        print(f"  Loaded {len(pretrained)} pretrained parameter tensors")

    # Freeze backbone; only train projection layers + metabolite branch + head
    if config['use_images']:
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("  Backbone frozen.")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config['learning_rate'], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

    history = defaultdict(list)
    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config['early_stopping_patience'])

    for epoch in range(config['num_epochs_phase1']):
        tl, ta = train_epoch(model, train_loader, optimizer, criterion, class_weights, config)
        vr = eval_epoch(model, val_loader, config)
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_acc'].append(vr['acc'])

        if vr['acc'] > best_val_acc:
            best_val_acc, best_state = vr['acc'], model.state_dict().copy()

        if epoch % 10 == 0:
            print(f"  Ep {epoch:3d}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")

        if es(vr['acc']):
            print(f"  Early stop at epoch {epoch}")
            break

    # Evaluate on test set
    model.load_state_dict(best_state)
    test_res = eval_epoch_detailed(model, test_loader, test_ds.df, config)

    auc = test_res.get('auc')
    auc_str = f"{auc:.3f}" if auc is not None and not np.isnan(float(auc)) else "N/A"
    print(f"\n  Result — Val: {best_val_acc:.3f}, Test Acc: {test_res['acc']:.3f}, AUC: {auc_str}")

    # Save outputs
    day_dir = output_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, day_dir / 'model.pth')

    pd.DataFrame(test_res['organoid_predictions']).to_csv(
        day_dir / 'organoid_predictions.csv', index=False)

    cm = test_res['confusion_matrix']
    with open(day_dir / 'metrics_test.json', 'w') as f:
        json.dump({
            'day':              day,
            'test_acc':         float(test_res['acc']),
            'test_f1':          float(test_res['f1']),
            'test_recall':      float(test_res['recall']),
            'test_precision':   float(test_res['precision']),
            'test_specificity': float(test_res['specificity']),
            'test_auc':         float(test_res['auc']) if test_res['auc'] else None,
            'test_pr_auc':      float(test_res['pr_auc']) if test_res['pr_auc'] else None,
            'test_acc_opt':     float(test_res['acc_opt']),
            'test_f1_opt':      float(test_res['f1_opt']),
            'opt_thresh':       float(test_res['opt_thresh']),
            'val_acc':          float(best_val_acc),
            'confusion_matrix': cm,
        }, f, indent=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss']); ax1.set_title(f'{day} - Loss'); ax1.set_xlabel('Epoch')
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'],   label='Val')
    ax2.set_title(f'{day} - Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend()
    plt.tight_layout()
    plt.savefig(day_dir / 'training_curves.png', dpi=150)
    plt.close()

    return {
        'day':              day,
        'day_no':           day_to_int(day),
        'val_acc':          best_val_acc,
        'test_acc':         test_res['acc'],
        'test_f1':          test_res['f1'],
        'test_recall':      test_res['recall'],
        'test_precision':   test_res['precision'],
        'test_specificity': test_res['specificity'],
        'test_auc':         test_res['auc'],
        'confusion_matrix': cm,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multimodal Organoid Classification')

    # Model
    parser.add_argument('--backbone', choices=['vit', 'resnet', 'efficientnet'], default='vit')
    parser.add_argument('--input-mode', choices=['rgb', 'overlay', 'rgb_mask', 'overlay_mask'], default='rgb')
    parser.add_argument('--fusion-strategy', choices=['concat', 'gated'], default='concat')
    parser.add_argument('--proj-dim', type=int, default=128,
                        help='Shared projection dimension for image and metabolite branches')

    # Modalities
    parser.add_argument('--use-images',      action='store_true', default=True)
    parser.add_argument('--use-metabolites', action='store_true', default=False)
    parser.add_argument('--images-only',     action='store_true')
    parser.add_argument('--metabolites-only', action='store_true')

    # Training
    parser.add_argument('--batch-size',              type=int,   default=16)
    parser.add_argument('--learning-rate',           type=float, default=1e-3)
    parser.add_argument('--num-epochs-phase1',       type=int,   default=50)
    parser.add_argument('--num-epochs-phase2',       type=int,   default=100)
    parser.add_argument('--early-stopping-patience', type=int,   default=20)
    parser.add_argument('--use-augmentation',        action='store_true')
    parser.add_argument('--use-growth-features',     action='store_true',
                        help='Append delta features (concentration change from previous timepoint). '
                             'Doubles metabolite dims: 4→8 (days ≤10) or 5→10 (days >10).')

    # Days
    parser.add_argument('--days', nargs='*', default=None,
                        help='Specific days to train (e.g., Dy03 Dy06). Defaults to all days.')

    # Paths
    parser.add_argument('--train-split', default='data/data_splits/both_train_base.json')
    parser.add_argument('--val-split',   default='data/data_splits/both_val_base.json')
    parser.add_argument('--test-split',  default='data/data_splits/both_test_base.json')
    parser.add_argument('--output-dir',  default='analysis/multimodal/outputs_multimodal')

    args = parser.parse_args()

    if args.images_only:
        args.use_images, args.use_metabolites = True, False
    elif args.metabolites_only:
        args.use_images, args.use_metabolites = False, True
    elif not args.use_metabolites:
        args.use_metabolites = False

    config = {
        'backbone':                args.backbone,
        'input_mode':              args.input_mode,
        'fusion_strategy':         args.fusion_strategy,
        'proj_dim':                args.proj_dim,
        'use_images':              args.use_images,
        'use_metabolites':         args.use_metabolites,
        'batch_size':              args.batch_size,
        'learning_rate':           args.learning_rate,
        'num_epochs_phase1':       args.num_epochs_phase1,
        'num_epochs_phase2':       args.num_epochs_phase2,
        'early_stopping_patience': args.early_stopping_patience,
        'target_size':             (384, 512),
        'use_augmentation':        args.use_augmentation,
        'use_growth_features':     args.use_growth_features,
        'train_split_path':        args.train_split,
        'val_split_path':          args.val_split,
        'test_split_path':         args.test_split,
        'device':                  'cuda' if torch.cuda.is_available() else 'cpu',
    }

    set_seed()

    print("\n" + "=" * 70)
    print("MULTIMODAL EXPERIMENT CONFIGURATION")
    print("=" * 70)
    for k, v in config.items():
        print(f"  {k:<30s}: {v}")
    print("=" * 70 + "\n")

    print("Loading data splits...")
    train_df, val_df, test_df = load_and_prepare_data(config)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    shared_backbone_state, shared_scaler = pretrain_shared_backbone(train_df, val_df, config)

    days_to_train = args.days or sorted(train_df['day'].unique(), key=day_to_int)
    print(f"Days to train: {days_to_train}\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for day in days_to_train:
        res = train_for_day(day, train_df, val_df, test_df, config, output_dir,
                            shared_backbone_state, shared_scaler)
        if res:
            results[day] = res

    if not results:
        print("No results to summarize.")
        return

    # Build summary
    rows = []
    for day, res in results.items():
        cm = res['confusion_matrix']
        rows.append({
            'Day': day, 'Day_No': res['day_no'],
            'Backbone': config['backbone'],
            'Test_Accuracy': res['test_acc'],
            'Test_F1': res['test_f1'],
            'Test_Recall': res['test_recall'],
            'Test_Precision': res['test_precision'],
            'Test_Specificity': res['test_specificity'],
            'Test_ROC_AUC': res['test_auc'],
            'TP': cm['TP'], 'FP': cm['FP'], 'TN': cm['TN'], 'FN': cm['FN'],
        })
    summary = pd.DataFrame(rows).sort_values('Day_No')

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(summary.to_string(index=False))

    summary.to_csv(output_dir / 'results_summary.csv', index=False)

    # Master CSV across runs
    model_id = f"{config['backbone']}_{config['input_mode']}_{config['fusion_strategy']}"
    summary = summary.assign(
        Model_ID=model_id,
        Input_Mode=config['input_mode'],
        Fusion_Strategy=config['fusion_strategy'],
        Use_Metabolites=config['use_metabolites'],
    )
    col_order = ['Model_ID', 'Backbone', 'Input_Mode', 'Fusion_Strategy', 'Use_Metabolites',
                 'Day', 'Day_No', 'Test_Accuracy', 'Test_F1', 'Test_Recall', 'Test_Precision',
                 'Test_Specificity', 'Test_ROC_AUC', 'TP', 'FP', 'TN', 'FN']
    summary = summary[col_order]

    overall_dir = output_dir.parent / 'overall'
    overall_dir.mkdir(parents=True, exist_ok=True)
    master_path = overall_dir / 'master_results.csv'
    if master_path.exists():
        existing = pd.read_csv(master_path)
        existing = existing[existing['Model_ID'] != model_id]
        master_df = pd.concat([existing, summary], ignore_index=True)
    else:
        master_df = summary
    master_df.sort_values(['Model_ID', 'Day_No']).to_csv(master_path, index=False)
    print(f"Master results updated at {master_path}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(summary['Day_No'], summary['Test_Accuracy'], 'o-')
    axes[0].set_title('Test Accuracy by Day'); axes[0].set_xlabel('Day'); axes[0].grid(alpha=0.3)
    axes[1].plot(summary['Day_No'], summary['Test_F1'], 'o-', color='orange')
    axes[1].set_title('Test F1 by Day'); axes[1].set_xlabel('Day'); axes[1].grid(alpha=0.3)
    if summary['Test_ROC_AUC'].notna().any():
        axes[2].plot(summary['Day_No'], summary['Test_ROC_AUC'], 'o-', color='green')
        axes[2].set_title('Test ROC-AUC by Day'); axes[2].set_xlabel('Day'); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_day.png', dpi=150)
    plt.close()

    print(f"\nResults saved to {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
