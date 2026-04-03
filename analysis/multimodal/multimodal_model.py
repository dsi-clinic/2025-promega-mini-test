"""
Multimodal organoid quality classification — model and data components.

Contains:
  - Constants (feature names, backbone registry)
  - OrganoidDataset
  - MaskBranch, MetaboliteBranch, MultimodalClassifier
  - EarlyStopping
  - get_transforms
  - load_and_prepare_data
  - train_epoch, eval_epoch, eval_epoch_detailed
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, confusion_matrix, recall_score, precision_score,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 1

# Metabolite features — NEVER use *_initial_concentration fields
BASE_MET_FEATURES = [
    'GlucoseGlo_concentration_uM',
    'GlutamateGlo_concentration_uM',
    'LactateGlo_concentration_uM',
    'PyruvateGlo_concentration_uM',
]
MALATE_FEATURE = 'MalateGlo_concentration_uM'  # only for days > 10

# Growth (delta) features — concentration change from the previous timepoint.
# Mirrors the logic in analysis/metabolites/classifier/train_metabolites.py.
# NaN on the first timepoint per organoid is filled with 0.
BASE_GROWTH_FEATURES = [
    'GlucoseGlo_growth',
    'GlutamateGlo_growth',
    'LactateGlo_growth',
    'PyruvateGlo_growth',
]
MALATE_GROWTH_FEATURE = 'MalateGlo_growth'  # only for days > 10

# Maximum metabolite input dimensions
# Without growth: 4 (days ≤10) or 5 (days >10)
# With    growth: 8 (days ≤10) or 10 (days >10)
MAX_META_DIM_BASE   = 5
MAX_META_DIM_GROWTH = 10

BACKBONE_MODELS = {
    'vit': 'vit_base_patch16_224',
    'resnet': 'resnet50',
    'efficientnet': 'efficientnet_b0',
}

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def day_to_int(day_str: str) -> int:
    m = re.search(r'[Dd][Yy](\d+)', day_str)
    return int(m.group(1)) if m else -1


def compute_growth_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add delta (growth) features: concentration change from the previous timepoint.

    Mirrors analysis/metabolites/classifier/train_metabolites.py.
    Rows are sorted by organoid and day before diffing, then NaNs
    (first timepoint per organoid) are filled with 0.

    New columns added:
      GlucoseGlo_growth, GlutamateGlo_growth, LactateGlo_growth,
      PyruvateGlo_growth, MalateGlo_growth
    """
    df = df.copy().sort_values(['org_id', 'day_num'])
    for base, growth in [
        ('GlucoseGlo_concentration_uM',   'GlucoseGlo_growth'),
        ('GlutamateGlo_concentration_uM', 'GlutamateGlo_growth'),
        ('LactateGlo_concentration_uM',   'LactateGlo_growth'),
        ('PyruvateGlo_concentration_uM',  'PyruvateGlo_growth'),
        ('MalateGlo_concentration_uM',    'MalateGlo_growth'),
    ]:
        df[growth] = df.groupby('org_id')[base].diff().fillna(0.0)
    return df


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class OrganoidDataset(Dataset):
    """PyTorch dataset that serves images, masks, and/or metabolite features."""

    def __init__(self, df, config, transform=None, scaler=None, fit_scaler=False):
        self.df = df.reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.label_map = {'Accepted': 1, 'Acceptable': 1, 'Not Accepted': 0, 'Not Acceptable': 0}
        self.img_key = 'overlay_path' if 'overlay' in config['input_mode'] else 'img_path'
        self.use_mask = 'mask' in config['input_mode']
        self.use_metabolites = config['use_metabolites']

        # Drop rows where required files are missing
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

        # Metabolite scaling
        if self.use_metabolites:
            self.meta_features_list = self._extract_metabolite_features()

            if fit_scaler:
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                self.scaler = StandardScaler()
                self.meta_features = self.scaler.fit_transform(np.array(padded, dtype=np.float32))
            elif scaler is not None:
                scaler_dim = scaler.mean_.shape[0]
                padded = [f + [0.0] * (scaler_dim - len(f)) for f in self.meta_features_list]
                self.scaler = scaler
                self.meta_features = self.scaler.transform(np.array(padded, dtype=np.float32))
            else:
                max_dim = max(len(f) for f in self.meta_features_list) if self.meta_features_list else 5
                padded = [f + [0.0] * (max_dim - len(f)) for f in self.meta_features_list]
                self.scaler = None
                self.meta_features = np.array(padded, dtype=np.float32)
        else:
            self.scaler = scaler

    def _extract_metabolite_features(self):
        """Return list of variable-length feature vectors.

        Without growth features (default):
          days ≤10 → 4 dims, days >10 → 5 dims

        With growth features (config['use_growth_features'] = True):
          days ≤10 → 8 dims  (4 base + 4 growth)
          days >10 → 10 dims (5 base + 5 growth)
        """
        use_growth = self.config.get('use_growth_features', False)
        features = []
        self.meta_dims = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            day_num = day_to_int(row.get('day', 'Dy00'))

            met_names = BASE_MET_FEATURES.copy()
            if day_num > 10:
                met_names.append(MALATE_FEATURE)

            if use_growth:
                met_names += BASE_GROWTH_FEATURES.copy()
                if day_num > 10:
                    met_names.append(MALATE_GROWTH_FEATURE)

            feat = [0.0 if pd.isna(row.get(n, np.nan)) else float(row.get(n)) for n in met_names]
            features.append(feat)
            self.meta_dims.append(len(feat))
        return features

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = torch.tensor(self.label_map.get(row['label'], 0), dtype=torch.float32)
        items = []

        if self.config['use_images']:
            img = Image.open(row[self.img_key]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            items.append(img)
            if self.use_mask:
                mask = Image.open(row['mask_path']).convert('L')
                mask = T.Compose([T.Resize(self.config['target_size']), T.ToTensor()])(mask)
                items.append(mask)

        if self.use_metabolites:
            items.append(torch.tensor(self.meta_features[idx], dtype=torch.float32))

        items.append(label)
        return tuple(items)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class MaskBranch(nn.Module):
    """Compact CNN encoder for binary segmentation masks."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 7, 2, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(), nn.Linear(32 * 16, out_dim), nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.encoder(x)


class MetaboliteBranch(nn.Module):
    """MLP that encodes 4-5 metabolite concentrations into a fixed-size vector.

    Outputs proj_dim to match the image projection for balanced fusion.
    """

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, proj_dim: int = 128):
        super().__init__()
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
        )
        self.out_dim = proj_dim

    def forward(self, x):
        if x.shape[1] < self.input_dim:
            padding = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device)
            x = torch.cat([x, padding], dim=1)
        return self.net(x)


class MultimodalClassifier(nn.Module):
    """Image + metabolite classifier with symmetric projection and configurable fusion.

    Both modalities are projected to proj_dim before fusion so neither branch
    dominates (image backbones typically produce 1024–2048-d while raw metabolite
    vectors are only 4–5 values).
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.use_mask = 'mask' in config['input_mode']
        self.use_metabolites = config['use_metabolites']
        self.fusion_strategy = config.get('fusion_strategy', 'concat')
        proj_dim = config.get('proj_dim', 128)

        # --- Image backbone ---
        backbone_name = BACKBONE_MODELS[config['backbone']]
        extra = {'img_size': config['target_size']} if 'vit' in backbone_name else {}
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, **extra)
        img_dim = self.backbone.num_features

        # Optional mask branch (appended to backbone output before projection)
        if self.use_mask:
            self.mask_branch = MaskBranch(64)
            img_dim += self.mask_branch.out_dim
        else:
            self.mask_branch = None

        # Project image features to proj_dim (symmetric with metabolite branch)
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # --- Metabolite branch ---
        if self.use_metabolites:
            use_growth = config.get('use_growth_features', False)
            meta_input_dim = MAX_META_DIM_GROWTH if use_growth else MAX_META_DIM_BASE
            self.meta_branch = MetaboliteBranch(input_dim=meta_input_dim, hidden_dim=64, proj_dim=proj_dim)
            if self.fusion_strategy == 'gated':
                self.gate = nn.Sequential(nn.Linear(proj_dim, proj_dim), nn.Sigmoid())
                fused_dim = proj_dim
            else:  # concat
                fused_dim = proj_dim * 2
        else:
            self.meta_branch = None
            fused_dim = proj_dim

        # --- Classification head ---
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, *args):
        # Parse inputs
        if self.config['use_images'] and self.use_metabolites:
            (img, mask, meta) = args[:3] if self.use_mask else (*args[:2], None)
            if not self.use_mask:
                img, meta = args[0], args[1]
        elif self.config['use_images']:
            img, mask = (args[0], args[1]) if self.use_mask else (args[0], None)
        else:
            meta = args[0]

        # Image path
        if self.config['use_images']:
            feats = self.backbone(img)
            if self.use_mask:
                feats = torch.cat([feats, self.mask_branch(mask)], dim=1)
            img_feats = self.img_proj(feats)

        # Metabolite path
        if self.use_metabolites:
            meta_feats = self.meta_branch(meta)

        # Fusion
        if self.config['use_images'] and self.use_metabolites:
            if self.fusion_strategy == 'gated':
                fused = img_feats * self.gate(meta_feats)
            else:
                fused = torch.cat([img_feats, meta_feats], dim=1)
        elif self.config['use_images']:
            fused = img_feats
        else:
            fused = meta_feats

        return self.head(fused).squeeze(1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best = -np.inf
        self.counter = 0

    def __call__(self, score: float) -> bool:
        if score > self.best + 1e-4:
            self.best, self.counter = score, 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def get_transforms(config: dict, augment: bool = False) -> T.Compose:
    t = [T.Resize(config['target_size'])]
    if augment and config['use_augmentation']:
        t.extend([T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.5)])
    t.extend([T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    return T.Compose(t)


def load_and_prepare_data(config: dict):
    """Load train/val/test split JSONs and return three DataFrames."""

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
                    'overlay_path': tp.get('overlay_path'),
                }
                metabolites = tp.get('metabolites', {})
                for name in BASE_MET_FEATURES:
                    row[name] = metabolites.get(name, np.nan)
                row[MALATE_FEATURE] = metabolites.get(MALATE_FEATURE, np.nan)
                rows.append(row)
        return pd.DataFrame(rows)

    train_df = json_to_df(load_json(config['train_split_path']))
    val_df   = json_to_df(load_json(config['val_split_path']))
    test_df  = json_to_df(load_json(config['test_split_path']))

    if config.get('use_growth_features', False):
        train_df = compute_growth_features(train_df)
        val_df   = compute_growth_features(val_df)
        test_df  = compute_growth_features(test_df)

    return train_df, val_df, test_df


def _unpack_batch(batch, config):
    """Separate model inputs from labels regardless of modality combination."""
    *inputs, y = batch
    return inputs, y


def train_epoch(model, loader, optimizer, criterion, class_weights, config):
    """Run one training epoch; return (mean_loss, accuracy)."""
    model.train()
    losses, preds, labels = [], [], []

    for batch in loader:
        inputs, y = _unpack_batch(batch, config)
        inputs = [x.to(config['device']) for x in inputs]
        y = y.to(config['device'])

        logits = model(*inputs)
        loss = criterion(logits, y)
        w = torch.tensor([class_weights[int(l)] for l in y], device=y.device)
        loss = (loss * w).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logits).detach().cpu().numpy())
        labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, (np.array(preds) > 0.5).astype(int))
    return float(np.mean(losses)), acc


def eval_epoch(model, loader, config):
    """Evaluate model; return metrics dict (no per-organoid breakdown)."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, y = _unpack_batch(batch, config)
            inputs = [x.to(config['device']) for x in inputs]
            preds.extend(torch.sigmoid(model(*inputs)).cpu().numpy())
            labels.extend(y.numpy())

    return _compute_metrics(np.array(preds), np.array(labels))


def eval_epoch_detailed(model, loader, dataset_df, config):
    """Evaluate model; return metrics dict including per-organoid predictions."""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch in loader:
            inputs, y = _unpack_batch(batch, config)
            inputs = [x.to(config['device']) for x in inputs]
            preds.extend(torch.sigmoid(model(*inputs)).cpu().numpy())
            labels.extend(y.numpy())

    preds, labels = np.array(preds), np.array(labels)
    metrics = _compute_metrics(preds, labels)

    preds_bin = (preds > 0.5).astype(int)
    organoid_results = []
    for idx in range(len(dataset_df)):
        true_label = int(labels[idx])
        pred_label = int(preds_bin[idx])
        if true_label == 1 and pred_label == 1:
            cat = 'TP'
        elif true_label == 0 and pred_label == 1:
            cat = 'FP'
        elif true_label == 1 and pred_label == 0:
            cat = 'FN'
        else:
            cat = 'TN'
        organoid_results.append({
            'Organoid_ID': dataset_df.iloc[idx]['org_id'],
            'True_Label': true_label,
            'Predicted_Probability': float(preds[idx]),
            'Predicted_Label': pred_label,
            'Correct': pred_label == true_label,
            'CM_Category': cat,
        })

    metrics['organoid_predictions'] = organoid_results
    return metrics


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    preds_bin = (preds > 0.5).astype(int)
    cm = confusion_matrix(labels, preds_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    opt_thresh = 0.5
    acc_opt = accuracy_score(labels, preds_bin)
    f1_opt  = f1_score(labels, preds_bin, zero_division=0)
    if len(np.unique(labels)) > 1:
        fpr, tpr, thresholds = roc_curve(labels, preds)
        idx = np.argmax(tpr - fpr)
        opt_thresh = float(thresholds[idx])
        preds_opt = (preds >= opt_thresh).astype(int)
        acc_opt = accuracy_score(labels, preds_opt)
        f1_opt  = f1_score(labels, preds_opt, zero_division=0)

    recall = recall_score(labels, preds_bin, zero_division=0)
    balanced_acc = (recall + specificity) / 2.0

    return {
        'acc':          accuracy_score(labels, preds_bin),
        'balanced_acc': balanced_acc,
        'f1':           f1_score(labels, preds_bin, zero_division=0),
        'recall':       recall,
        'precision':    precision_score(labels, preds_bin, zero_division=0),
        'specificity':  specificity,
        'auc':        roc_auc_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'pr_auc':     average_precision_score(labels, preds) if len(np.unique(labels)) > 1 else None,
        'acc_opt':    acc_opt,
        'f1_opt':     f1_opt,
        'opt_thresh': opt_thresh,
        'preds':      preds,
        'labels':     labels,
        'confusion_matrix': {'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)},
    }
