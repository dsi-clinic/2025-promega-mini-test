#!/usr/bin/env python3

import os, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import timm
from torchvision import transforms as T

# -------- Config --------
BACKBONES = {
    #"resnet": "resnet50",
    "efficientnet": "efficientnet_b0"
}
DATA_DIR = Path("data/preprocessed/majority/")
OUT_ROOT = Path("outputs_(256,192)")
BATCH_SIZE = 16
TARGET_SIZE = (256,192)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
# ------------------------

class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = -np.inf
        self.bad = 0

    def step(self, score):
        if score > self.best + self.min_delta:
            self.best = score
            self.bad = 0
            return False
        self.bad += 1
        return self.bad >= self.patience

class OrganoidDataset(Dataset):
    def __init__(self, img_paths, mask_paths, labels, augment=False):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.augment = augment
        self.t_img = T.Compose([T.Resize(TARGET_SIZE), T.ToTensor()])
        self.t_mask = T.Compose([T.Resize(TARGET_SIZE, interpolation=T.InterpolationMode.NEAREST), T.ToTensor()])
        self.cj = T.ColorJitter(0.2, 0.2, 0.2, 0.1)

    def __len__(self): return len(self.labels)

    def _flip(self, img, mask):
        if torch.rand(()) > 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
        return img, mask

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        img, mask = self.t_img(img), self.t_mask(mask)
        if self.augment:
            img, mask = self._flip(img, mask)
            img = self.cj(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, mask, label

class OrganoidClassifier(nn.Module):
    def __init__(self, backbone_name):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=True, num_classes=0, global_pool="avg")
        self.backbone_out = self.backbone.num_features

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (TARGET_SIZE[0]//4) * (TARGET_SIZE[1]//4), 64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out + 64, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def unfreeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if "blocks." in name or "layer" in name:
                param.requires_grad = True

    def forward(self, img, mask):
        f_img = self.backbone(img)
        f_mask = self.mask_branch(mask)
        x = torch.cat([f_img, f_mask], dim=1)
        return self.classifier(x).squeeze(1)

def make_loader(imgs, masks, labels, augment, batch_size):
    ds = OrganoidDataset(imgs, masks, labels, augment)
    return DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=NUM_WORKERS)

def epoch_loop(model, loader, optimizer, weights, train=True):
    model.train() if train else model.eval()
    bce = nn.BCEWithLogitsLoss(reduction="none")
    losses, preds, trues = [], [], []

    for img, mask, label in loader:
        img, mask, label = img.to(DEVICE), mask.to(DEVICE), label.to(DEVICE)
        logit = model(img, mask)
        loss = bce(logit, label)
        weight = torch.tensor([weights[int(l.item())] for l in label], device=label.device)
        loss = (loss * weight).mean()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
        trues.extend(label.cpu().numpy())

    preds_bin = (np.array(preds) > 0.5).astype(int)
    f1 = f1_score(trues, preds_bin, average="weighted")
    return np.mean(losses), f1

def run_training(data_path, backbone_key, backbone_name):
    records = json.loads(data_path.read_text())
    if not records:
        print(f"⚠ Skipping {data_path.name} — no records")
        return

    label_map = {"Accepted": 1, "Not Accepted": 0}
    imgs = np.array([r["img_path"] for r in records])
    masks = np.array([r["mask_path"] for r in records])
    labels = np.array([label_map[r["label"]] for r in records])

    X_tr, X_val, M_tr, M_val, y_tr, y_val = train_test_split(
        imgs, masks, labels, test_size=0.2, stratify=labels, random_state=42)

    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_weights = {int(k): float(w) for k, w in zip(np.unique(y_tr), weights)}

    train_loader = make_loader(X_tr, M_tr, y_tr, augment=True, batch_size=BATCH_SIZE)
    val_loader = make_loader(X_val, M_val, y_val, augment=False, batch_size=BATCH_SIZE)

    model = OrganoidClassifier(backbone_name).to(DEVICE)
    model_dir = OUT_ROOT / backbone_key / data_path.stem
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pth"

    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    history = defaultdict(list)
    best_f1 = -np.inf

    # Phase 1 — frozen
    for epoch in range(100):
        tl, tf1 = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vf1 = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_f1"].append(tf1)
        history["val_f1"].append(vf1)
        print(f"[{data_path.stem}][{backbone_key}][P1][{epoch:02d}] loss {tl:.4f}/{vl:.4f} f1 {tf1:.3f}/{vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), model_path)
        if es.step(vf1):
            break

    # Phase 2 — unfreeze
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for epoch in range(300):
        tl, tf1 = epoch_loop(model, train_loader, opt, class_weights, train=True)
        vl, vf1 = epoch_loop(model, val_loader, opt, class_weights, train=False)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_f1"].append(tf1)
        history["val_f1"].append(vf1)
        print(f"[{data_path.stem}][{backbone_key}][P2][{epoch:03d}] loss {tl:.4f}/{vl:.4f} f1 {tf1:.3f}/{vf1:.3f}")
        if vf1 > best_f1:
            best_f1 = vf1
            torch.save(model.state_dict(), model_path)
        if es.step(vf1):
            break

    # Save plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.plot(history["train_f1"]); plt.plot(history["val_f1"]); plt.title("Weighted F1")
    plt.subplot(1, 2, 2); plt.plot(history["train_loss"]); plt.plot(history["val_loss"]); plt.title("Loss")
    plt.savefig(model_dir / "training_curves.png")
    print(f"📈 Saved curves → {model_dir/'training_curves.png'}")

    # Confusion matrix
    model.load_state_dict(torch.load(model_path))
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for img, mask, lbl in val_loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)
            prob = torch.sigmoid(model(img, mask)).cpu().numpy()
            preds.extend((prob > 0.5).astype(int))
            trues.extend(lbl.numpy())

    cm = confusion_matrix(trues, preds)
    final_f1 = f1_score(trues, preds, average="weighted")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Accept", "Accept"],
                yticklabels=["Not Accept", "Accept"])
    plt.title(f"Confusion Matrix (F1: {final_f1:.3f})")
    plt.savefig(model_dir / "confusion_matrix.png")
    print(f"📊 Saved confusion matrix → {model_dir/'confusion_matrix.png'}")
    print(f"✅ Final validation weighted-F1: {final_f1:.4f}")


def main():
    for backbone_key, backbone_name in BACKBONES.items():
        for json_file in sorted(DATA_DIR.glob("Dy*.json")):
            run_training(json_file, backbone_key, backbone_name)

if __name__ == "__main__":
    main()
