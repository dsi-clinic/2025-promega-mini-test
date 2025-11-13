#!/usr/bin/env python3
"""
Multimodal Organoid Quality Classification
Combines image and metabolite data for prediction using configurable fusion strategies.
"""

import os
import json
import re
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, average_precision_score, 
                             roc_curve, confusion_matrix, recall_score, precision_score)

# Configuration
SEED = 42

# Metabolite features - match Meta_classifier_new_split.ipynb exactly
# NEVER use *_initial_concentration fields
BASE_MET_FEATURES = [
    'GlucoseGlo_concentration_uM',
    'GlutamateGlo_concentration_uM',
    'LactateGlo_concentration_uM',
    'PyruvateGlo_concentration_uM'
]
MALATE_FEATURE = 'MalateGlo_concentration_uM'  # Only included for days >10

BACKBONE_MODELS = {
    'vit': 'vit_base_patch16_224',
    'resnet': 'resnet50',
    'efficientnet': 'efficientnet_b0'
}

def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def day_to_int(day_str):
    m = re.search(r'[Dd][Yy](\d+)', day_str)
    return int(m.group(1)) if m else -1

class OrganoidDataset(Dataset):
    """Dataset with images and metabolites."""
    
    def __init__(self, df, config, transform=None, scaler=None, fit_scaler=False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.label_map = {'Accepted': 1, 'Acceptable': 1, 'Not Accepted': 0, 'Not Acceptable': 0}
        self.img_key = 'overlay_path' if 'overlay' in config['input_mode'] else 'img_path'
        self.use_mask = 'mask' in config['input_mode']
        self.use_metabolites = config['use_metabolites']
        
        # Filter valid samples
        valid = []
        for i in range(len(self.df)):
            if config['use_images']:
                img = self.df.iloc[i][self.img_key]
                if pd.isna(img) or not Path(img).exists():
                    continue
                if self.use_mask:
                    mask = self.df.iloc[i]['mask_path']
                    if pd.isna(mask) or not Path(mask).exists():
                        continue
            valid.append(i)
        
        self.df = self.df.iloc[valid].reset_index(drop=True)
        
        # Extract and scale metabolite features
        if self.use_metabolites:
            self.meta_features_list = self._extract_metabolite_features()  # list of variable-length arrays
            
            # Pad to maximum dimension before converting to numpy array
            # This handles mixed days (4-dim for days ≤10, 5-dim for days >10)
            if fit_scaler:
                # Fit scaler: pad all vectors to max dimension
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded_features = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                
                self.scaler = StandardScaler()
                self.meta_features = np.array(padded_features, dtype=np.float32)
                self.meta_features = self.scaler.fit_transform(self.meta_features)
            elif scaler is not None:
                # Use existing scaler: pad to match scaler's expected dimension
                scaler_dim = scaler.mean_.shape[0]
                padded_features = [f + [0.0] * (scaler_dim - len(f)) for f in self.meta_features_list]
                
                self.scaler = scaler
                self.meta_features = np.array(padded_features, dtype=np.float32)
                self.meta_features = self.scaler.transform(self.meta_features)
            else:
                # No scaler: pad to max dimension
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded_features = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                
                self.scaler = None
                self.meta_features = np.array(padded_features, dtype=np.float32)
        else:
            self.scaler = scaler
    
    def _extract_metabolite_features(self):
        """
        Extract metabolite features - match Meta_classifier_new_split.ipynb exactly.
        Returns list of variable-length feature vectors:
        - Days ≤10: 4 features (no Malate)
        - Days >10: 5 features (includes Malate)
        """
        features = []
        self.meta_dims = []  # Track per-sample dimensionality
        
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            day_str = row.get('day', 'Dy00')
            day_num = day_to_int(day_str)
            
            # Build day-aware feature list (NO initial_concentration fields)
            met_names = BASE_MET_FEATURES.copy()
            if day_num > 10:
                met_names.append(MALATE_FEATURE)
            
            # Extract values
            feat = []
            for met_name in met_names:
                val = row.get(met_name, np.nan)
                feat.append(0.0 if pd.isna(val) else float(val))
            
            features.append(feat)
            self.meta_dims.append(len(feat))
        
        return features  # Return as list-of-lists (variable length)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.label_map.get(row['label'], 0), dtype=torch.float32)
        
        items = []
        
        # Add image
        if self.config['use_images']:
            img = Image.open(row[self.img_key]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            items.append(img)
            
            if self.use_mask:
                mask = Image.open(row['mask_path']).convert('L')
                mask = T.Compose([T.Resize(self.config['target_size']), T.ToTensor()])(mask)
                items.append(mask)
        
        # Add metabolites
        if self.use_metabolites:
            meta = torch.tensor(self.meta_features[idx], dtype=torch.float32)
            items.append(meta)
        
        items.append(label)
        return tuple(items)

class MaskBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(), nn.Linear(32*16, out_dim), nn.ReLU()
        )
        self.out_dim = out_dim
    
    def forward(self, x):
        return self.encoder(x)

class MetaboliteBranch(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        """
        Metabolite branch MLP.
        input_dim: Max possible dimension (5 for days >10, but will handle 4 for days ≤10)
        """
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_dim = hidden_dim
    
    def forward(self, x):
        # x may have variable width depending on day
        # If input is smaller than expected, pad with zeros
        if x.shape[1] < self.input_dim:
            padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        return self.net(x)

class MultimodalClassifier(nn.Module):
    """Image + Metabolite classifier with configurable fusion."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_mask = 'mask' in config['input_mode']
        self.use_metabolites = config['use_metabolites']
        self.fusion_strategy = config.get('fusion_strategy', 'concat')  # 'concat' or 'gated'
        
        # Image backbone
        backbone_name = BACKBONE_MODELS[config['backbone']]
        extra = {'img_size': config['target_size']} if 'vit' in backbone_name else {}
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, **extra)
        # Do NOT freeze here - will be frozen later in train_for_day
        
        img_dim = self.backbone.num_features
        
        # Mask branch
        if self.use_mask:
            self.mask_branch = MaskBranch(64)
            img_dim += self.mask_branch.out_dim
        else:
            self.mask_branch = None
        
        # Metabolite branch
        if self.use_metabolites:
            self.meta_branch = MetaboliteBranch(input_dim=5, hidden_dim=64)
            meta_dim = self.meta_branch.out_dim
            
            if self.fusion_strategy == 'gated':
                # Gated fusion: metabolite modulates image features
                self.gate = nn.Sequential(
                    nn.Linear(meta_dim, img_dim),
                    nn.Sigmoid()
                )
                fused_dim = img_dim
            else:
                # Concat fusion
                fused_dim = img_dim + meta_dim
        else:
            fused_dim = img_dim
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, *args):
        """Forward pass handling different input combinations."""
        # Parse inputs based on configuration
        if self.config['use_images'] and self.use_metabolites:
            if self.use_mask:
                img, mask, meta = args[:3]
            else:
                img, meta = args[:2]
        elif self.config['use_images']:
            if self.use_mask:
                img, mask = args[:2]
            else:
                img = args[0]
        elif self.use_metabolites:
            meta = args[0]
        else:
            raise ValueError("Must use either images or metabolites")
        
        # Extract image features
        if self.config['use_images']:
            img_feats = self.backbone(img)
            if self.use_mask:
                img_feats = torch.cat([img_feats, self.mask_branch(mask)], 1)
        
        # Extract metabolite features
        if self.use_metabolites:
            meta_feats = self.meta_branch(meta)
        
        # Fusion
        if self.config['use_images'] and self.use_metabolites:
            if self.fusion_strategy == 'gated':
                gate = self.gate(meta_feats)
                fused = img_feats * gate
            else:  # concat
                fused = torch.cat([img_feats, meta_feats], 1)
        elif self.config['use_images']:
            fused = img_feats
        else:
            fused = meta_feats
        
        return self.head(fused).squeeze(1)

class EarlyStopping:
    def __init__(self, patience=20):
        self.patience, self.best, self.counter = patience, -np.inf, 0
    
    def __call__(self, score):
        if score > self.best + 1e-4:
            self.best, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience

def get_transforms(config, augment=False):
    t = [T.Resize(config['target_size'])]
    if augment and config['use_augmentation']:
        t.extend([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)])
    t.extend([T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
    return T.Compose(t)

def load_and_prepare_data(config):
    """Load split data and convert to DataFrames with metabolite features."""
    def load_json(path):
        with open(path) as f:
            return json.load(f)
    
    def json_to_df(data):
        rows = []
        for org_id, info in data.items():
            for day, tp in info.get('timepoints', {}).items():
                row = {
                    'org_id': org_id,
                    'label': info.get('label'),
                    'day': day,
                    'day_num': day_to_int(day),
                    'img_path': tp.get('img_path'),
                    'mask_path': tp.get('mask_path'),
                    'overlay_path': tp.get('overlay_path')
                }
                # Add metabolite features - only concentration, NO initial_concentration
                metabolites = tp.get('metabolites', {})
                for met_name in BASE_MET_FEATURES:
                    row[met_name] = metabolites.get(met_name, np.nan)
                # Add Malate separately (will be included based on day in dataset)
                row[MALATE_FEATURE] = metabolites.get(MALATE_FEATURE, np.nan)
                rows.append(row)
        return pd.DataFrame(rows)
    
    train_df = json_to_df(load_json(config['train_split_path']))
    val_df = json_to_df(load_json(config['val_split_path']))
    test_df = json_to_df(load_json(config['test_split_path']))
    
    return train_df, val_df, test_df

def train_epoch(model, loader, opt, crit, weights, config):
    model.train()
    losses, preds, labels = [], [], []
    
    for batch in loader:
        # Unpack based on modality
        if config['use_images'] and config['use_metabolites']:
            if 'mask' in config['input_mode']:
                *inputs, y = batch  # img, mask, meta, label
            else:
                *inputs, y = batch  # img, meta, label
        elif config['use_images']:
            if 'mask' in config['input_mode']:
                *inputs, y = batch  # img, mask, label
            else:
                *inputs, y = batch  # img, label
        else:
            *inputs, y = batch  # meta, label
        
        inputs = [x.to(config['device']) for x in inputs]
        y = y.to(config['device'])
        
        logits = model(*inputs)
        loss = crit(logits, y)
        w = torch.tensor([weights[int(l)] for l in y], device=y.device)
        loss = (loss * w).mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())
    
    acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
    return np.mean(losses), acc

def eval_epoch(model, loader, config):
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            if config['use_images'] and config['use_metabolites']:
                if 'mask' in config['input_mode']:
                    *inputs, y = batch
                else:
                    *inputs, y = batch
            elif config['use_images']:
                if 'mask' in config['input_mode']:
                    *inputs, y = batch
                else:
                    *inputs, y = batch
            else:
                *inputs, y = batch
            
            inputs = [x.to(config['device']) for x in inputs]
            probs = torch.sigmoid(model(*inputs)).cpu().numpy()
            preds.extend(probs)
            labels.extend(y.numpy())
    
    preds, labels = np.array(preds), np.array(labels)
    preds_bin = (preds > 0.5).astype(int)
    
    # Compute confusion matrix for 0.5 threshold
    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate Specificity / True Negative Rate (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate standard metrics
    recall = recall_score(labels, preds_bin, zero_division=0)
    precision = precision_score(labels, preds_bin, zero_division=0)
    
    # Compute ROC-based optimal threshold
    opt_thresh = 0.5
    acc_opt = accuracy_score(labels, preds_bin)
    f1_opt = f1_score(labels, preds_bin, zero_division=0)
    
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, preds)
        optimal_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[optimal_idx]
        preds_opt = (preds >= opt_thresh).astype(int)
        acc_opt = accuracy_score(labels, preds_opt)
        f1_opt = f1_score(labels, preds_opt, zero_division=0)
    
    return {
        'acc': accuracy_score(labels, preds_bin),
        'f1': f1_score(labels, preds_bin, zero_division=0),
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'pr_auc': average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'acc_opt': acc_opt,
        'f1_opt': f1_opt,
        'opt_thresh': opt_thresh,
        'preds': preds,
        'labels': labels,
        'confusion_matrix': {
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        }
    }

def eval_epoch_detailed(model, loader, dataset_df, config):
    """
    Evaluate model and return per-organoid predictions.
    
    Args:
        model: The trained model
        loader: DataLoader for evaluation
        dataset_df: DataFrame containing organoid IDs
        config: Configuration dictionary
    
    Returns:
        Dictionary with metrics and per-organoid predictions
    """
    model.eval()
    preds, labels = [], []
    
    with torch.no_grad():
        for batch in loader:
            if config['use_images'] and config['use_metabolites']:
                if 'mask' in config['input_mode']:
                    *inputs, y = batch
                else:
                    *inputs, y = batch
            elif config['use_images']:
                if 'mask' in config['input_mode']:
                    *inputs, y = batch
                else:
                    *inputs, y = batch
            else:
                *inputs, y = batch
            
            inputs = [x.to(config['device']) for x in inputs]
            probs = torch.sigmoid(model(*inputs)).cpu().numpy()
            preds.extend(probs)
            labels.extend(y.numpy())
    
    preds, labels = np.array(preds), np.array(labels)
    preds_bin = (preds > 0.5).astype(int)
    
    # Create per-organoid results
    organoid_results = []
    for idx in range(len(dataset_df)):
        org_id = dataset_df.iloc[idx]['org_id']
        true_label = int(labels[idx])
        pred_prob = float(preds[idx])
        pred_label = int(preds_bin[idx])
        correct = (pred_label == true_label)
        
        # Determine confusion matrix category
        if true_label == 1 and pred_label == 1:
            cm_category = 'TP'
        elif true_label == 0 and pred_label == 1:
            cm_category = 'FP'
        elif true_label == 1 and pred_label == 0:
            cm_category = 'FN'
        else:  # true_label == 0 and pred_label == 0
            cm_category = 'TN'
        
        organoid_results.append({
            'Organoid_ID': org_id,
            'True_Label': true_label,
            'Predicted_Probability': pred_prob,
            'Predicted_Label': pred_label,
            'Correct': correct,
            'CM_Category': cm_category
        })
    
    # Compute confusion matrix for 0.5 threshold
    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # Calculate Specificity / True Negative Rate (TNR)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate standard metrics
    recall = recall_score(labels, preds_bin, zero_division=0)
    precision = precision_score(labels, preds_bin, zero_division=0)
    
    # Compute ROC-based optimal threshold
    opt_thresh = 0.5
    acc_opt = accuracy_score(labels, preds_bin)
    f1_opt = f1_score(labels, preds_bin, zero_division=0)
    
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, preds)
        optimal_idx = np.argmax(tpr - fpr)
        opt_thresh = thresholds[optimal_idx]
        preds_opt = (preds >= opt_thresh).astype(int)
        acc_opt = accuracy_score(labels, preds_opt)
        f1_opt = f1_score(labels, preds_opt, zero_division=0)
    
    return {
        'acc': accuracy_score(labels, preds_bin),
        'f1': f1_score(labels, preds_bin, zero_division=0),
        'recall': recall,
        'precision': precision,
        'specificity': specificity,
        'auc': roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'pr_auc': average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'acc_opt': acc_opt,
        'f1_opt': f1_opt,
        'opt_thresh': opt_thresh,
        'preds': preds,
        'labels': labels,
        'confusion_matrix': {
            'TP': int(tp),
            'FP': int(fp),
            'TN': int(tn),
            'FN': int(fn)
        },
        'organoid_predictions': organoid_results
    }

def pretrain_shared_backbone(train_df, val_df, config):
    """
    Pretrain a shared backbone using all training samples from all days combined.
    Returns the best backbone state_dict and the global metabolite scaler.
    """
    print(f"\n{'='*60}\nPretraining Shared Backbone (All Days)\n{'='*60}")
    
    # Transforms
    t_train = get_transforms(config, augment=True) if config['use_images'] else None
    t_eval = get_transforms(config, augment=False) if config['use_images'] else None
    
    # Create datasets with all days combined
    train_ds = OrganoidDataset(train_df, config, t_train, fit_scaler=True)
    scaler = train_ds.scaler
    val_ds = OrganoidDataset(val_df, config, t_eval, scaler=scaler)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=4)
    
    # Class weights (use filtered dataset)
    labels = [train_ds.label_map.get(train_ds.df.iloc[i]['label'], 0) for i in range(len(train_ds))]
    weights_arr = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(labels), weights_arr)}
    
    # Model
    model = MultimodalClassifier(config).to(config['device'])
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    crit = nn.BCEWithLogitsLoss(reduction='none')
    
    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config['early_stopping_patience'])
    
    print(f"Training shared backbone (all parameters trainable)")
    for epoch in range(config['num_epochs_phase1']):
        tl, ta = train_epoch(model, train_loader, opt, crit, class_weights, config)
        vr = eval_epoch(model, val_loader, config)
        
        if vr['acc'] > best_val_acc:
            best_val_acc, best_state = vr['acc'], model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f"Ep {epoch}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")
        
        if es(vr['acc']):
            print(f"Early stop at epoch {epoch}")
            break
    
    print(f"Shared backbone pretraining complete. Best val acc: {best_val_acc:.3f}\n")
    
    return best_state, scaler


def train_for_day(day, train_df, val_df, test_df, config, output_dir,
                  shared_backbone_state=None, shared_scaler=None):
    """Train model for a specific day."""
    print(f"\n{'='*60}\nTraining for {day}\n{'='*60}")
    
    train_day = train_df[train_df['day'] == day]
    val_day = val_df[val_df['day'] == day]
    test_day = test_df[test_df['day'] == day]
    
    if len(train_day) == 0:
        print(f"No training data for {day}")
        return None
    
    def label_counts(df):
        return df['label'].value_counts().to_dict()
    
    train_counts = label_counts(train_day)
    val_counts = label_counts(val_day)
    test_counts = label_counts(test_day)

    print(f"{day} label counts:")
    print(f"  Train: {train_counts}")
    print(f"  Val  : {val_counts}")
    print(f"  Test : {test_counts}")
    
    # Transforms
    t_train = get_transforms(config, augment=True) if config['use_images'] else None
    t_eval = get_transforms(config, augment=False) if config['use_images'] else None
    
    # Datasets - use global scaler if provided, otherwise fit on day-specific data
    if shared_scaler is not None:
        train_ds = OrganoidDataset(train_day, config, t_train, scaler=shared_scaler)
        scaler = shared_scaler
    else:
        train_ds = OrganoidDataset(train_day, config, t_train, fit_scaler=True)
        scaler = train_ds.scaler
    
    val_ds = OrganoidDataset(val_day, config, t_eval, scaler=scaler)
    test_ds = OrganoidDataset(test_day, config, t_eval, scaler=scaler)
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], num_workers=4)
    
    # Class weights
    labels = [train_ds.label_map.get(train_day.iloc[i]['label'], 0) for i in range(len(train_day))]
    weights_arr = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = {int(c): float(w) for c, w in zip(np.unique(labels), weights_arr)}
    
    # Model
    model = MultimodalClassifier(config).to(config['device'])
    
    # Load shared backbone weights if provided
    if shared_backbone_state is not None:
        print("Loading shared backbone weights...")
        # Load matching keys for backbone and meta_branch
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in shared_backbone_state.items() 
                          if k in model_dict and (k.startswith('backbone.') or k.startswith('meta_branch.'))}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} pretrained parameters")
    
    # Freeze backbone for all days
    if config['use_images']:
        print("Freezing backbone...")
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=1e-4
    )
    crit = nn.BCEWithLogitsLoss(reduction='none')
    
    history = defaultdict(list)
    best_val_acc, best_state = -np.inf, None
    es = EarlyStopping(config['early_stopping_patience'])
    
    # Train day-specific classifier head (backbone remains frozen)
    print(f"Training day-specific head (backbone frozen)")
    for epoch in range(config['num_epochs_phase1']):
        tl, ta = train_epoch(model, train_loader, opt, crit, class_weights, config)
        vr = eval_epoch(model, val_loader, config)
        history['train_loss'].append(tl)
        history['train_acc'].append(ta)
        history['val_acc'].append(vr['acc'])
        
        if vr['acc'] > best_val_acc:
            best_val_acc, best_state = vr['acc'], model.state_dict().copy()
        
        if epoch % 10 == 0:
            print(f"Ep {epoch}: loss={tl:.4f}, tr_acc={ta:.3f}, val_acc={vr['acc']:.3f}")
        
        if es(vr['acc']):
            print(f"Early stop at epoch {epoch}")
            break
    
    # Evaluate best model with detailed per-organoid tracking
    model.load_state_dict(best_state)
    test_res = eval_epoch_detailed(model, test_loader, test_ds.df, config)
    
    auc = test_res.get('auc')
    try:
        auc_f = float(auc); auc_str = "N/A" if np.isnan(auc_f) else f"{auc_f:.3f}"
    except (TypeError, ValueError):
        auc_str = "N/A"
    print(f"\nFinal - Val: {best_val_acc:.3f}, Test Acc: {test_res['acc']:.3f}, AUC: {auc_str}")
    
    # Save results
    day_dir = output_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    torch.save(best_state, day_dir / 'model.pth')
    
    # Save per-organoid predictions to CSV
    organoid_preds_df = pd.DataFrame(test_res['organoid_predictions'])
    organoid_preds_df.to_csv(day_dir / 'organoid_predictions.csv', index=False)
    print(f"Saved per-organoid predictions to {day_dir / 'organoid_predictions.csv'}")
    
    # Save metrics including ROC-optimal threshold metrics, confusion matrix, and specificity
    cm = test_res['confusion_matrix']
    with open(day_dir / 'metrics_test.json', 'w') as f:
        json.dump({
            'day': day,
            'test_acc': float(test_res['acc']),
            'test_f1': float(test_res['f1']),
            'test_recall': float(test_res['recall']),
            'test_precision': float(test_res['precision']),
            'test_specificity': float(test_res['specificity']),
            'test_auc': float(test_res['auc']) if test_res['auc'] else None,
            'test_pr_auc': float(test_res['pr_auc']) if test_res['pr_auc'] else None,
            'test_acc_opt': float(test_res['acc_opt']),
            'test_f1_opt': float(test_res['f1_opt']),
            'opt_thresh': float(test_res['opt_thresh']),
            'val_acc': float(best_val_acc),
            'confusion_matrix': {
                'TP': cm['TP'],
                'FP': cm['FP'],
                'TN': cm['TN'],
                'FN': cm['FN']
            }
        }, f, indent=2)
    
    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'])
    ax1.set_title(f'{day} - Loss')
    ax1.set_xlabel('Epoch')
    
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Val')
    ax2.set_title(f'{day} - Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(day_dir / 'training_curves.png', dpi=150)
    plt.close()
    
    return {
        'day': day,
        'day_no': day_to_int(day),
        'val_acc': best_val_acc,
        'test_acc': test_res['acc'],
        'test_f1': test_res['f1'],
        'test_recall': test_res['recall'],
        'test_precision': test_res['precision'],
        'test_specificity': test_res['specificity'],
        'test_auc': test_res['auc'],
        'confusion_matrix': test_res['confusion_matrix']
    }

def main():
    parser = argparse.ArgumentParser(description='Multimodal Organoid Classification')
    
    # Model selection
    parser.add_argument('--backbone', choices=['vit', 'resnet', 'efficientnet'], default='vit')
    parser.add_argument('--input-mode', choices=['rgb', 'overlay', 'rgb_mask', 'overlay_mask'], default='rgb')
    parser.add_argument('--fusion-strategy', choices=['concat', 'gated'], default='concat',
                       help='Fusion strategy: concat or gated (metabolite modulates image)')
    
    # Modality selection  
    parser.add_argument('--use-images', action='store_true', default=True)
    parser.add_argument('--use-metabolites', action='store_true', default=False)
    parser.add_argument('--images-only', action='store_true', help='Use only images (no metabolites)')
    parser.add_argument('--metabolites-only', action='store_true', help='Use only metabolites (no images)')
    
    # Training params
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--num-epochs-phase1', type=int, default=50)
    parser.add_argument('--num-epochs-phase2', type=int, default=100)
    parser.add_argument('--early-stopping-patience', type=int, default=20)
    parser.add_argument('--use-augmentation', action='store_true')
    
    # Days to train
    parser.add_argument('--days', nargs='*', default=None, help='Specific days to train (e.g., Dy03 Dy06)')
    
    # Paths
    parser.add_argument('--train-split', default='data_splits/both_train_base.json')
    parser.add_argument('--val-split', default='data_splits/both_val_base.json')
    parser.add_argument('--test-split', default='data_splits/both_test_base.json')
    parser.add_argument('--output-dir', default='analysis/multimodal/outputs_multimodal')
    
    args = parser.parse_args()
    
    # Handle modality flags
    if args.images_only:
        args.use_images = True
        args.use_metabolites = False
    elif args.metabolites_only:
        args.use_images = False
        args.use_metabolites = True
    elif not args.use_metabolites:
        args.use_metabolites = False  # Default to images only unless --use-metabolites
    
    config = {
        'backbone': args.backbone,
        'input_mode': args.input_mode,
        'fusion_strategy': args.fusion_strategy,
        'use_images': args.use_images,
        'use_metabolites': args.use_metabolites,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs_phase1': args.num_epochs_phase1,
        'num_epochs_phase2': args.num_epochs_phase2,
        'early_stopping_patience': args.early_stopping_patience,
        'target_size': (384, 512),
        'use_augmentation': args.use_augmentation,
        'train_split_path': args.train_split,
        'val_split_path': args.val_split,
        'test_split_path': args.test_split,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    set_seed()
    
    print("\n" + "="*70)
    print("MULTIMODAL EXPERIMENT CONFIGURATION")
    print("="*70)
    for k, v in config.items():
        print(f"{k:30s}: {v}")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data splits...")
    train_df, val_df, test_df = load_and_prepare_data(config)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Pretrain shared backbone using all days
    shared_backbone_state, shared_scaler = pretrain_shared_backbone(train_df, val_df, config)
    
    # Determine days to train
    if args.days:
        days_to_train = args.days
    else:
        days_to_train = sorted(train_df['day'].unique(), key=day_to_int)
    
    print(f"Days to train: {days_to_train}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train per day with shared backbone
    results = {}
    for day in days_to_train:
        res = train_for_day(day, train_df, val_df, test_df, config, output_dir,
                           shared_backbone_state, shared_scaler)
        if res:
            results[day] = res
    
    # Summary
    if results:
        # Create comprehensive summary with all metrics including confusion matrix
        summary_rows = []
        for day, res in results.items():
            cm = res['confusion_matrix']
            summary_rows.append({
                'Day': day,
                'Day_No': res['day_no'],
                'Backbone': config['backbone'],
                'Test_Accuracy': res['test_acc'],
                'Test_F1': res['test_f1'],
                'Test_Recall': res['test_recall'],
                'Test_Precision': res['test_precision'],
                'Test_Specificity': res['test_specificity'],
                'Test_ROC_AUC': res['test_auc'] if res['test_auc'] else None,
                'TP': cm['TP'],
                'FP': cm['FP'],
                'TN': cm['TN'],
                'FN': cm['FN']
            })
        
        summary = pd.DataFrame(summary_rows).sort_values('Day_No')
        
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        print(summary.to_string(index=False))
        
        # Save summary for this model configuration
        summary.to_csv(output_dir / 'results_summary.csv', index=False)
        
        # Create/update master CSV in outputs_multimodal/overall/
        overall_dir = output_dir.parent / 'overall'
        overall_dir.mkdir(parents=True, exist_ok=True)
        master_csv_path = overall_dir / 'master_results.csv'
        
        # Add model configuration identifier
        model_id = f"{config['backbone']}_{config['input_mode']}_{config['fusion_strategy']}"
        summary['Model_ID'] = model_id
        summary['Input_Mode'] = config['input_mode']
        summary['Fusion_Strategy'] = config['fusion_strategy']
        summary['Use_Metabolites'] = config['use_metabolites']
        
        # Reorder columns for clarity
        col_order = ['Model_ID', 'Backbone', 'Input_Mode', 'Fusion_Strategy', 'Use_Metabolites', 
                     'Day', 'Day_No', 'Test_Accuracy', 'Test_F1', 'Test_Recall', 'Test_Precision', 
                     'Test_Specificity', 'Test_ROC_AUC', 'TP', 'FP', 'TN', 'FN']
        summary = summary[col_order]
        
        # Append or create master CSV
        if master_csv_path.exists():
            existing_df = pd.read_csv(master_csv_path)
            # Remove any existing entries for this model configuration
            existing_df = existing_df[existing_df['Model_ID'] != model_id]
            # Append new results
            master_df = pd.concat([existing_df, summary], ignore_index=True)
        else:
            master_df = summary
        
        # Sort by Model_ID and Day_No
        master_df = master_df.sort_values(['Model_ID', 'Day_No'])
        master_df.to_csv(master_csv_path, index=False)
        
        print(f"\nMaster results updated at {master_csv_path}")
        
        # Plot metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        axes[0].plot(summary['Day_No'], summary['Test_Accuracy'], 'o-')
        axes[0].set_title('Test Accuracy by Day')
        axes[0].set_xlabel('Day')
        axes[0].set_ylabel('Accuracy')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(summary['Day_No'], summary['Test_F1'], 'o-', color='orange')
        axes[1].set_title('Test F1 by Day')
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('F1 Score')
        axes[1].grid(True, alpha=0.3)
        
        if summary['Test_ROC_AUC'].notna().any():
            axes[2].plot(summary['Day_No'], summary['Test_ROC_AUC'], 'o-', color='green')
            axes[2].set_title('Test ROC-AUC by Day')
            axes[2].set_xlabel('Day')
            axes[2].set_ylabel('ROC-AUC')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metrics_by_day.png', dpi=150)
        plt.close()
        
        print(f"\nResults saved to {output_dir}")
        print("="*70)

if __name__ == '__main__':
    main()
