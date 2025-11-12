#!/usr/bin/env python3

import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T
import timm
from transformers import AutoModel
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
import re

# Configuration
TARGET_SIZE = (384, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_BASE_DIR = Path("/home/tonyluo/minitest/analysis/images/classifier")
DATA_DIR = Path("/net/projects2/promega/data-analysis/output")

BACKBONES = {
    "dinov2": "facebook/dinov2-base",
    "resnet": "resnet50",
    "efficientnet": "efficientnet_b0"
}

# Data splits mapping
DATA_SPLITS = {
    "img_path_nomask_ALL": {
        "output_dir": "outputs_img_path_nomask_all",
        "train_split": "data_splits/both_train_base.json",
        "val_split": "data_splits/both_val_base.json",
        "test_split": "data_splits/both_test_base.json",
        "input_key": "img_path",
        "use_mask": False
    },
    "img_path_nomask_CLEAN": {
        "output_dir": "outputs_img_path_nomask_clean",
        "train_split": "data_splits/both_train_base_clean.json",
        "val_split": "data_splits/both_val_base_clean.json",
        "test_split": "data_splits/both_test_base_clean.json",
        "input_key": "img_path",
        "use_mask": False
    },
    "img_path_mask_ALL": {
        "output_dir": "outputs_img_path_mask_all",
        "train_split": "data_splits/both_train_base.json",
        "val_split": "data_splits/both_val_base.json",
        "test_split": "data_splits/both_test_base.json",
        "input_key": "img_path",
        "use_mask": True
    },
    "img_path_mask_CLEAN": {
        "output_dir": "outputs_img_path_mask_clean",
        "train_split": "data_splits/both_train_base_clean.json",
        "val_split": "data_splits/both_val_base_clean.json",
        "test_split": "data_splits/both_test_base_clean.json",
        "input_key": "img_path",
        "use_mask": True
    },
    "overlay_path_nomask_ALL": {
        "output_dir": "outputs_overlay_path_nomask_all",
        "train_split": "data_splits/both_train_base.json",
        "val_split": "data_splits/both_val_base.json",
        "test_split": "data_splits/both_test_base.json",
        "input_key": "overlay_path",
        "use_mask": False
    },
    "overlay_path_nomask_CLEAN": {
        "output_dir": "outputs_overlay_path_nomask_clean",
        "train_split": "data_splits/both_train_base_clean.json",
        "val_split": "data_splits/both_val_base_clean.json",
        "test_split": "data_splits/both_test_base_clean.json",
        "input_key": "overlay_path",
        "use_mask": False
    },
    "overlay_path_mask_ALL": {
        "output_dir": "outputs_overlay_path_mask_all",
        "train_split": "data_splits/both_train_base.json",
        "val_split": "data_splits/both_val_base.json",
        "test_split": "data_splits/both_test_base.json",
        "input_key": "overlay_path",
        "use_mask": True
    },
    "overlay_path_mask_CLEAN": {
        "output_dir": "outputs_overlay_path_mask_clean",
        "train_split": "data_splits/both_train_base_clean.json",
        "val_split": "data_splits/both_val_base_clean.json",
        "test_split": "data_splits/both_test_base_clean.json",
        "input_key": "overlay_path",
        "use_mask": True
    }
}

# Model classes (same as train_model_multimodal.py)
class MaskBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.out_dim = out_dim
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, out_dim)
        
    def forward(self, mask):
        x = self.conv_layers(mask)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class ImageOnlyClassifier(nn.Module):
    def __init__(self, backbone_key, backbone_name, target_size, use_mask=False):
        super().__init__()
        self.use_mask = use_mask
        self.backbone_key = backbone_key
        self._is_dinov2 = (backbone_key == "dinov2")

        if self._is_dinov2:
            self.backbone = AutoModel.from_pretrained(backbone_name, local_files_only=True)
            out_dim = self.backbone.config.hidden_size
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg"
            )
            out_dim = self.backbone.num_features

        if self.use_mask:
            self.mask_branch = MaskBranch(out_dim=64)
            head_in = out_dim + self.mask_branch.out_dim
        else:
            self.mask_branch = None
            head_in = out_dim

        self.classifier = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, img, mask=None):
        if self._is_dinov2:
            outputs = self.backbone(img)
            f = outputs.last_hidden_state[:, 0, :]
        else:
            f = self.backbone(img)

        if self.use_mask and mask is not None:
            mask_f = self.mask_branch(mask)
            f = torch.cat([f, mask_f], dim=1)

        return self.classifier(f).squeeze(1)

def day_to_int(day_str):
    """Convert day string (e.g., 'Dy03') to integer"""
    match = re.search(r'(\d+(?:\.\d+)?)', day_str)
    if match:
        return float(match.group(1))
    return 0.0

def is_normal_image(img_path):
    """Check if image is normal (nosplit_nostitch)"""
    img_str = str(img_path).lower()
    return "nosplit_nostitch" in img_str

def classify_image_type(img_path):
    """Classify image as: normal, split, stitched, or split_and_stitched"""
    img_str = str(img_path).lower()
    
    has_split = "split" in img_str and "nosplit" not in img_str
    has_stitch = "stitch" in img_str and "nostitch" not in img_str
    
    if has_split and has_stitch:
        return "split_and_stitched"
    elif has_split:
        return "split"
    elif has_stitch:
        return "stitched"
    else:
        return "normal"

def resolve_input_path(timepoint, input_key):
    """Resolve input path, handling overlay_path fallback"""
    if input_key == "overlay_path":
        overlay_path = timepoint.get("overlay_path")
        if overlay_path and Path(overlay_path).exists():
            return overlay_path
        
        # Try to construct from mask_path
        mask_path = timepoint.get("mask_path")
        if mask_path:
            overlay_candidate = mask_path.replace("predicted_masks", "image_mask_overlays")
            overlay_candidate = overlay_candidate.replace("_predmask", "_overlay")
            if Path(overlay_candidate).exists():
                return overlay_candidate
        return None
    else:
        return timepoint.get(input_key)

def load_model(config_name, backbone_key):
    """Load trained model"""
    config = DATA_SPLITS[config_name]
    model_path = RESULTS_BASE_DIR / config["output_dir"] / backbone_key / "model.pth"
    
    if not model_path.exists():
        return None
    
    backbone_name = BACKBONES[backbone_key]
    model = ImageOnlyClassifier(backbone_key, backbone_name, TARGET_SIZE, config["use_mask"]).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=False))
    model.eval()
    return model

def get_transforms(backbone_key, use_mask):
    """Get image and mask transforms"""
    if backbone_key == "dinov2":
        img_transform = T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        img_transform = T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    mask_transform = T.Compose([
        T.Resize(TARGET_SIZE),
        T.ToTensor()
    ]) if use_mask else None
    
    return img_transform, mask_transform

def evaluate_day_backbone(config_name, backbone_key, test_data):
    """Evaluate model on test data, grouped by day"""
    print(f"  Evaluating {backbone_key}...")
    
    # Load model
    model = load_model(config_name, backbone_key)
    if model is None:
        print(f"    ⚠ Model not found, skipping")
        return None
    
    config = DATA_SPLITS[config_name]
    img_transform, mask_transform = get_transforms(backbone_key, config["use_mask"])
    
    # Group test data by day
    day_data = defaultdict(list)
    for organoid_id, organoid_data in test_data.items():
        label = 1 if organoid_data["label"] == "Acceptable" else 0
        for day, timepoint in organoid_data.get("timepoints", {}).items():
            input_path = resolve_input_path(timepoint, config["input_key"])
            if not input_path or not Path(input_path).exists():
                continue
            
            mask_path = timepoint.get("mask_path") if config["use_mask"] else None
            
            day_data[day].append({
                "input_path": input_path,
                "mask_path": mask_path,
                "label": label,
                "organoid_id": organoid_id
            })
    
    # Evaluate per day
    results = []
    for day in sorted(day_data.keys(), key=day_to_int):
        samples = day_data[day]
        if not samples:
            continue
        
        # Run inference
        all_labels = []
        all_preds = []
        all_probs = []
        image_types = []
        
        with torch.no_grad():
            for sample in samples:
                try:
                    # Load image
                    img = Image.open(sample["input_path"]).convert("RGB")
                    img_tensor = img_transform(img).unsqueeze(0).to(DEVICE)
                    
                    # Load mask if needed
                    mask_tensor = None
                    if config["use_mask"] and sample["mask_path"] and Path(sample["mask_path"]).exists():
                        mask = Image.open(sample["mask_path"]).convert("L")
                        mask_tensor = mask_transform(mask).unsqueeze(0).to(DEVICE)
                    elif config["use_mask"]:
                        mask_tensor = torch.zeros(1, 1, TARGET_SIZE[0], TARGET_SIZE[1]).to(DEVICE)
                    
                    # Predict
                    if mask_tensor is not None:
                        logit = model(img_tensor, mask_tensor)
                    else:
                        logit = model(img_tensor)
                    
                    prob = torch.sigmoid(logit).cpu().item()
                    pred = 1 if prob > 0.5 else 0
                    
                    all_labels.append(sample["label"])
                    all_preds.append(pred)
                    all_probs.append(prob)
                    image_types.append(classify_image_type(sample["input_path"]))
                    
                except Exception as e:
                    print(f"    ⚠ Error processing {sample['input_path']}: {e}")
                    continue
        
        if not all_labels:
            continue
        
        # Calculate metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC AUC
        if len(set(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, y_prob)
        else:
            roc_auc = 0.5
        
        # Count image types
        type_counts = defaultdict(int)
        for img_type in image_types:
            type_counts[img_type] += 1
        
        number_split = type_counts["split"]
        number_stitch = type_counts["stitched"]
        number_split_and_stitched = type_counts["split_and_stitched"]
        number_normal = type_counts["normal"]
        total = len(image_types)
        
        # Calculate percentages
        percent_split = (number_split / total * 100) if total > 0 else 0.0
        percent_stitched = (number_stitch / total * 100) if total > 0 else 0.0
        percent_split_and_stitched = (number_split_and_stitched / total * 100) if total > 0 else 0.0
        
        # Calculate error rates by image type
        error_rates = defaultdict(lambda: 0.0)
        for i, (img_type, true_label, pred_label) in enumerate(zip(image_types, y_true, y_pred)):
            if pred_label != true_label:
                error_rates[img_type] += 1
        
        error_rate_normal = (error_rates["normal"] / number_normal * 100) if number_normal > 0 else 0.0
        error_rate_split = (error_rates["split"] / number_split * 100) if number_split > 0 else 0.0
        error_rate_stitched = (error_rates["stitched"] / number_stitch * 100) if number_stitch > 0 else 0.0
        error_rate_split_and_stitched = (error_rates["split_and_stitched"] / number_split_and_stitched * 100) if number_split_and_stitched > 0 else 0.0
        
        # Get sample counts (load train/val splits)
        train_file = Path(DATA_SPLITS[config_name]["train_split"])
        val_file = Path(DATA_SPLITS[config_name]["val_split"])
        
        train_samples = 0
        val_samples = 0
        
        if train_file.exists():
            with open(train_file) as f:
                train_data = json.load(f)
            for org_data in train_data.values():
                if day in org_data.get("timepoints", {}):
                    train_samples += 1
        
        if val_file.exists():
            with open(val_file) as f:
                val_data = json.load(f)
            for org_data in val_data.values():
                if day in org_data.get("timepoints", {}):
                    val_samples += 1
        
        day_no = day_to_int(day)
        
        results.append({
            "Day": day,
            "Day_No": int(day_no) if day_no == int(day_no) else day_no,
            "Backbone": backbone_key,
            "Test_Accuracy": round(accuracy, 4),
            "Test_F1": round(f1, 4),
            "Test_Recall": round(recall, 4),
            "Test_Precision": round(precision, 4),
            "Test_ROC_AUC": round(roc_auc, 4),
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Train_Samples": train_samples,
            "Val_Samples": val_samples,
            "Test_Samples": total,
            "Number_Split": number_split,
            "Number_Stitch": number_stitch,
            "Percent_Split_%": round(percent_split, 2),
            "Percent_Stitched_%": round(percent_stitched, 2),
            "Percent_Split_And_Stitched_%": round(percent_split_and_stitched, 2),
            "Error_Rate_Normal_Images_%": round(error_rate_normal, 2),
            "Error_Rate_Split_Images_%": round(error_rate_split, 2),
            "Error_Rate_Stitched_Images_%": round(error_rate_stitched, 2),
            "Error_Rate_Split_And_Stitched_Images_%": round(error_rate_split_and_stitched, 2)
        })
    
    return results

def generate_summary_table(config_name):
    """Generate summary table for one configuration"""
    print(f"\n{'='*80}")
    print(f"Generating summary table for: {config_name}")
    print(f"{'='*80}")
    
    config = DATA_SPLITS[config_name]
    test_file = Path(config["test_split"])
    
    if not test_file.exists():
        print(f"  ⚠ Test split not found: {test_file}")
        return None
    
    # Load test data
    with open(test_file) as f:
        test_data = json.load(f)
    
    print(f"  Loaded {len(test_data)} test organoids")
    
    # Evaluate each backbone
    all_results = []
    for backbone_key in BACKBONES.keys():
        results = evaluate_day_backbone(config_name, backbone_key, test_data)
        if results:
            all_results.extend(results)
    
    if not all_results:
        print(f"  ⚠ No results generated")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Sort by Day_No, then Backbone
    df = df.sort_values(["Day_No", "Backbone"])
    
    # Save to home directory
    output_file = Path.home() / f"classifier_summary_table_{config_name.lower()}.csv"
    df.to_csv(output_file, index=False)
    
    print(f"  ✓ Saved {len(df)} rows to: {output_file}")
    
    return df

def main():
    print("="*80)
    print("GENERATING DETAILED SUMMARY TABLES FOR ALL 8 CONFIGURATIONS")
    print("="*80)
    print(f"Device: {DEVICE}")
    print()
    
    for config_name in DATA_SPLITS.keys():
        try:
            generate_summary_table(config_name)
        except Exception as e:
            print(f"  ❌ Error generating table for {config_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("DONE! All summary tables saved to home directory")
    print("="*80)

if __name__ == "__main__":
    main()

