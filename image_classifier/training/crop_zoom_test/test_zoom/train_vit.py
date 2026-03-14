#!/usr/bin/env python
# --------------------------------------------------------------
#  train_model_efficient.py • Dy30 majority dataset
#  EfficientNet-B0 backbone  + mask-CNN  + EarlyStopping
#  Now with **Random Zoom-in + Crop** augmentation                <-- NEW
# --------------------------------------------------------------
import json
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

import timm
from torchvision import transforms as T

# -------------------- constants --------------------
DATA_JSON = Path("data/preprocessed/majority/Dy30.json")
MODEL_OUT = Path("models/organoid_classifier_dy30_efficient_zoom_test.pth")
PLOT_DIR = Path("outputs_efficientnet_zoom/test_plots")
BATCH_SIZE = 16
TARGET_SIZE = (224, 224)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0  # increase if your system allows
# ---------------------------------------------------


# ---------------- EarlyStopping -------------------
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad = -np.inf, 0

    def step(self, metric):
        if metric > self.best + self.min_delta:
            self.best, self.bad = metric, 0
            return False
        self.bad += 1
        return self.bad >= self.patience


# ---------------------------------------------------


# ------------------ dataset ------------------------
class OrganoidDataset(Dataset):
    """
    Dataset that returns (img, mask, label).
    When `augment=True`, applies:
        • Random zoom-in + crop (prob. 0.5)         <-- NEW
        • Random horizontal flip
    """

    def __init__(
        self, img_paths, mask_paths, labels, augment=False, zoom_scale=(0.8, 1.0)
    ):  # <-- new arg
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.augment = augment
        self.zoom_scale = zoom_scale

        # separate “resize” and “to-tensor” so we can insert crop beforehand
        self.t_resize_img = T.Resize(TARGET_SIZE)
        self.t_resize_mask = T.Resize(
            TARGET_SIZE, interpolation=T.InterpolationMode.NEAREST
        )
        self.t_to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.labels)

    # -------- augmentation helpers ---------
    @staticmethod
    def _flip(img, mask):
        if torch.rand(()) > 0.5:
            img = T.functional.hflip(img)
            mask = T.functional.hflip(mask)
        return img, mask

    @staticmethod
    def _zoom_crop_pair(img, mask, scale=(0.8, 1.0)):
        """
        Apply the *same* RandomResizedCrop to an RGB image and its mask.
        Aspect-ratio locked to 1.0 ⇒ pure zoom-in/out (no stretching).
        """
        i, j, h, w = T.RandomResizedCrop.get_params(img, scale=scale, ratio=(1.0, 1.0))

        img = T.functional.resized_crop(
            img,
            i,
            j,
            h,
            w,
            size=TARGET_SIZE,
            interpolation=T.InterpolationMode.BILINEAR,
        )

        mask = T.functional.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            size=TARGET_SIZE,
            interpolation=T.InterpolationMode.NEAREST,
        )
        return img, mask

    # ---------------------------------------

    def __getitem__(self, idx):
        # ---- load PIL images ----
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # ---- random zoom-crop (p=0.5) ----
        if self.augment:  # and torch.rand(()) >= 0.5:
            img, mask = self._zoom_crop_pair(img, mask, scale=self.zoom_scale)
        else:
            # deterministic resize path
            img = self.t_resize_img(img)
            mask = self.t_resize_mask(mask)

        # ---- other augmentations ----
        if self.augment:
            img, mask = self._flip(img, mask)

        # ---- to tensor ----
        img = self.t_to_tensor(img)
        mask = self.t_to_tensor(mask)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, mask, label


# ---------------------------------------------------


# --------------- model definition ------------------
class OrganoidClassifierEff(nn.Module):
    def __init__(self, eff_name="efficientnet_b0", unfreeze_layers=5):
        super().__init__()
        self.backbone = timm.create_model(
            eff_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.backbone_out = self.backbone.num_features
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.unfreeze_layers = unfreeze_layers

        self.mask_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * (TARGET_SIZE[0] // 4) * (TARGET_SIZE[1] // 4), 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.backbone_out + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def unfreeze_backbone(self):
        # unfreeze last N layers (param order)
        for p in list(self.backbone.parameters())[-self.unfreeze_layers :]:
            p.requires_grad = True

    def forward(self, img, mask):
        feat_img = self.backbone(img)  # (B, feat)
        feat_mask = self.mask_branch(mask)
        x = torch.cat([feat_img, feat_mask], dim=1)
        return self.classifier(x).squeeze(1)


# ---------------------------------------------------


def make_loader(imgs, masks, labels, augment, batch, zoom_scale=(0.8, 1.0)):
    ds = OrganoidDataset(imgs, masks, labels, augment, zoom_scale=zoom_scale)
    return DataLoader(
        ds, batch_size=batch, shuffle=augment, num_workers=NUM_WORKERS, pin_memory=True
    )


def epoch_loop(model, loader, optim, weights, train=True):
    model.train() if train else model.eval()
    preds, trues, losses = [], [], []
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for img, mask, lbl in loader:
        img, mask, lbl = img.to(DEVICE), mask.to(DEVICE), lbl.to(DEVICE)
        logit = model(img, mask)
        loss = bce(logit, lbl)
        w = torch.tensor([weights[int(y.item())] for y in lbl], device=lbl.device)
        loss = (loss * w).mean()
        if train:
            optim.zero_grad()
            loss.backward()
            optim.step()
        losses.append(loss.item())
        preds.extend(torch.sigmoid(logit).detach().cpu().numpy())
        trues.extend(lbl.cpu().numpy())
    preds_bin = (np.array(preds) > 0.5).astype(int)
    f1 = f1_score(trues, preds_bin, average="weighted")
    return np.mean(losses), f1


def main():
    # ---------- load Dy30 majority ----------
    records = json.loads(DATA_JSON.read_text())
    label_map = {"Accepted": 1, "Not Accepted": 0}
    imgs = np.array([r["img_path"] for r in records])
    masks = np.array([r["mask_path"] for r in records])
    labels = np.array([label_map[r["label"]] for r in records])

    X_tr, X_val, M_tr, M_val, y_tr, y_val = train_test_split(
        imgs, masks, labels, test_size=0.2, stratify=labels, random_state=42
    )

    print(X_tr.shape, X_val.shape, M_tr.shape, M_val.shape, y_tr.shape, y_val.shape)

    cw_arr = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    class_w = {int(c): float(w) for c, w in zip(np.unique(y_tr), cw_arr)}

    train_loader = make_loader(
        X_tr, M_tr, y_tr, augment=False, batch=BATCH_SIZE, zoom_scale=(1.0, 1.0)
    )
    val_loader = make_loader(
        X_val, M_val, y_val, augment=True, batch=BATCH_SIZE, zoom_scale=(1.0, 1.0)
    )

    # ------------- model ---------------
    model = OrganoidClassifierEff().to(DEVICE)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    h, best = {"train_f1": [], "val_f1": [], "train_loss": [], "val_loss": []}, -np.inf

    # ----- phase-1 (frozen) -----
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    es = EarlyStopping(patience=20)
    for ep in range(1, 200):
        tl, tf1 = epoch_loop(model, train_loader, opt, class_w, True)
        vl, vf1 = epoch_loop(model, val_loader, opt, class_w, False)
        h["train_loss"].append(tl)
        h["val_loss"].append(vl)
        h["train_f1"].append(tf1)
        h["val_f1"].append(vf1)
        print(f"[P1][{ep:02d}] loss {tl:.4f}/{vl:.4f}  " f"f1 {tf1:.3f}/{vf1:.3f}")
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), MODEL_OUT)
        if es.step(vf1):
            print(f"--> ES P1 {ep}")
            break

    # ----- phase-2 (fine-tune) -----
    model.unfreeze_backbone()
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    es = EarlyStopping(patience=30)
    for ep in range(1, 300):
        tl, tf1 = epoch_loop(model, train_loader, opt, class_w, True)
        vl, vf1 = epoch_loop(model, val_loader, opt, class_w, False)
        h["train_loss"].append(tl)
        h["val_loss"].append(vl)
        h["train_f1"].append(tf1)
        h["val_f1"].append(vf1)
        print(f"[P2][{ep:03d}] loss {tl:.4f}/{vl:.4f}  " f"f1 {tf1:.3f}/{vf1:.3f}")
        if vf1 > best:
            best = vf1
            torch.save(model.state_dict(), MODEL_OUT)
        if es.step(vf1):
            print(f"--> ES P2 {ep}")
            break

    print(f"\nBest val F1: {best:.3f}  →  {MODEL_OUT}")

    # ------------- plots --------------
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(h["train_f1"], label="train")
    plt.plot(h["val_f1"], label="val")
    plt.title("Weighted F1")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(h["train_loss"], label="train")
    plt.plot(h["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()
    plt.savefig(PLOT_DIR / "training_curves_dy30_efficient.png")

    # ---------- confusion matrix -------
    model.load_state_dict(torch.load(MODEL_OUT))
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for img, mask, lbl in val_loader:
            probs = torch.sigmoid(model(img.to(DEVICE), mask.to(DEVICE))).cpu().numpy()
            preds.extend((probs > 0.5).astype(int))
            trues.extend(lbl.numpy())

    cm = confusion_matrix(trues, preds)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Accept", "Accept"],
        yticklabels=["Not Accept", "Accept"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Dy30, EfficientNet)")
    plt.savefig(PLOT_DIR / "confusion_matrix_dy30_efficient_80.png")
    final_f1 = f1_score(trues, preds, average="weighted")
    print(f"Final validation weighted-F1: {final_f1:.4f}")


if __name__ == "__main__":
    main()
