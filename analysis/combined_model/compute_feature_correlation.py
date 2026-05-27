#!/usr/bin/env python3
"""
Compute correlation between image embeddings and metabolite features.

Purpose:
  Test whether image morphology embeddings and metabolite features contain
  overlapping or complementary information for the combined model.

Uses:
  - Canonical splits
  - Current-day metabolite concentrations only
  - Pretrained EfficientNet-B0 image embeddings
  - PCA-reduced image embeddings
  - Canonical Correlation Analysis by day

Outputs:
  analysis/combined_model/outputs/feature_correlation/
    - aligned_features.csv
    - cca_by_day.csv
    - metabolite_pc_correlations.csv
    - cca_by_day.png
    - metabolite_pc_heatmap.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import timm

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

import matplotlib.pyplot as plt


PROJ_ROOT = Path(__file__).resolve().parents[2]

DAY_ORDER = [
    "Dy03", "Dy06", "Dy08", "Dy10", "Dy13",
    "Dy15", "Dy17", "Dy20_5", "Dy24", "Dy28", "Dy30",
]

BASE_MET_FEATURES = [
    "GlucoseGlo_concentration_uM",
    "GlutamateGlo_concentration_uM",
    "LactateGlo_concentration_uM",
    "PyruvateGlo_concentration_uM",
]

MALATE_FEATURE = "MalateGlo_concentration_uM"


def day_to_int(day):
    """Convert a day string like 'Dy03' or 'Dy20_5' to an integer day number.

    Args:
        day: Day string in format 'DyNN' or 'DyNN_N'.

    Returns:
        Integer day number.
    """
    if day == "Dy20_5":
        return 20
    return int(day.replace("Dy", "").split("_")[0])


def get_metabolite_columns(day):
    """Return the list of metabolite feature columns for a given day.

    Malate is only included for days after day 10.

    Args:
        day: Day string (e.g. 'Dy03', 'Dy20_5').

    Returns:
        List of metabolite column name strings.
    """
    cols = BASE_MET_FEATURES.copy()
    if day_to_int(day) > 10:
        cols.append(MALATE_FEATURE)
    return cols


class ImageEmbeddingDataset(Dataset):
    """PyTorch Dataset for loading organoid images for embedding extraction."""

    def __init__(self, df, image_col, transform):
        """Initialize the dataset.

        Args:
            df: DataFrame containing organoid records with image paths.
            image_col: Column name for image file paths.
            transform: Torchvision transform to apply to each image.
        """
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """Load and transform a single image and return it with metadata.

        Args:
            idx: Integer index of the sample.

        Returns:
            Dict with keys image, org_id, day, label, and split.
        """
        row = self.df.iloc[idx]
        path = row[self.image_col]

        img = Image.open(path).convert("RGB")
        img = self.transform(img)

        return {
            "image": img,
            "org_id": row["org_id"],
            "day": row["day"],
            "label": row["label"],
            "split": row["split"],
        }


def get_transform(target_size=(384, 512)):
    """Build a standard image preprocessing transform.

    Args:
        target_size: Tuple (height, width) to resize images to.

    Returns:
        A composed torchvision transform.
    """
    return T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_all_data(input_mode):
    """Load all organoid records into a DataFrame using canonical splits.

    Args:
        input_mode: Either 'rgb' or 'overlay' to select image type.

    Returns:
        DataFrame with one row per organoid-day record, including image paths,
        metabolite concentrations, label, and split assignment.
    """
    sys.path.insert(0, str(PROJ_ROOT))

    from pipeline.data_loader import OrganoidDataset, default_filters
    from pipeline.splits import Splits

    ds = OrganoidDataset(
        str(PROJ_ROOT / "data/all_data.json"),
        splits=Splits.canonical(),
        filters=default_filters(),
    )

    rows = []
    image_col = "overlay_path" if input_mode == "overlay" else "img_path"

    for org_id, info in ds.iter_organoids():
        for day, rec in info["records"].items():
            if day not in DAY_ORDER:
                continue

            imgs = rec.get("images") or {}
            mets = rec.get("metabolite") or {}

            row = {
                "org_id": org_id,
                "day": day,
                "label": info["label"],
                "split": info.get("split"),
                "img_path": imgs.get("img_path"),
                "overlay_path": imgs.get("overlay_path"),
            }

            for met_name in BASE_MET_FEATURES:
                key = met_name.replace("_concentration_uM", "")
                row[met_name] = (mets.get(key) or {}).get("concentration_uM", np.nan)

            malate_key = MALATE_FEATURE.replace("_concentration_uM", "")
            row[MALATE_FEATURE] = (mets.get(malate_key) or {}).get("concentration_uM", np.nan)

            if row.get(image_col) is None:
                continue

            if not Path(row[image_col]).exists():
                continue

            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"Loaded {len(df)} records")
    print(f"Unique organoids: {df['org_id'].nunique()}")
    print(df["split"].value_counts(dropna=False))
    print(df["label"].value_counts(dropna=False))

    return df


def extract_image_embeddings(df, input_mode, batch_size, device):
    """Extract EfficientNet-B0 embeddings for all images in the DataFrame.

    Args:
        df: DataFrame with image path columns.
        input_mode: Either 'rgb' or 'overlay' to select image column.
        batch_size: Number of images per batch.
        device: Torch device string ('cuda' or 'cpu').

    Returns:
        DataFrame with org_id, day, label, split, and one column per embedding dimension.
    """
    image_col = "overlay_path" if input_mode == "overlay" else "img_path"

    transform = get_transform()
    dataset = ImageEmbeddingDataset(df, image_col=image_col, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=0,
    ).to(device)

    model.eval()

    embeddings = []
    meta_rows = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            feats = model(images).cpu().numpy()

            embeddings.append(feats)

            batch_size_actual = feats.shape[0]
            for i in range(batch_size_actual):
                meta_rows.append({
                    "org_id": batch["org_id"][i],
                    "day": batch["day"][i],
                    "label": batch["label"][i],
                    "split": batch["split"][i],
                })

    emb = np.vstack(embeddings)
    meta = pd.DataFrame(meta_rows)

    emb_cols = [f"img_emb_{i}" for i in range(emb.shape[1])]
    emb_df = pd.DataFrame(emb, columns=emb_cols)

    return pd.concat([meta, emb_df], axis=1)


def prepare_aligned_features(df, emb_df):
    """Merge image embeddings with metabolite features on shared key columns.

    Args:
        df: Original records DataFrame with metabolite columns.
        emb_df: DataFrame of image embeddings with org_id, day, label, split columns.

    Returns:
        Merged DataFrame containing both embedding and metabolite columns.
    """
    key_cols = ["org_id", "day", "label", "split"]

    met_cols_all = BASE_MET_FEATURES + [MALATE_FEATURE]
    met_df = df[key_cols + met_cols_all].copy()

    aligned = emb_df.merge(
        met_df,
        on=key_cols,
        how="inner",
    )

    return aligned


def compute_cca_by_day(aligned, output_dir, pca_dim):
    """Compute Canonical Correlation Analysis between image PCs and metabolites per day.

    Args:
        aligned: DataFrame with both image embedding and metabolite columns.
        output_dir: Path to save cca_by_day.csv.
        pca_dim: Number of PCA components to reduce image embeddings to before CCA.

    Returns:
        DataFrame with CCA results per day.
    """
    results = []
    img_cols = [c for c in aligned.columns if c.startswith("img_emb_")]

    for day in DAY_ORDER:
        day_df = aligned[aligned["day"] == day].copy()

        if len(day_df) < 10:
            print(f"Skipping {day}: too few samples")
            continue

        met_cols = get_metabolite_columns(day)
        day_df = day_df.dropna(subset=met_cols)

        if len(day_df) < 10:
            print(f"Skipping {day}: too few complete metabolite samples")
            continue

        X_img = day_df[img_cols].values
        X_met = day_df[met_cols].values

        n_samples = len(day_df)
        n_met = X_met.shape[1]

        img_scaled = StandardScaler().fit_transform(X_img)
        met_scaled = StandardScaler().fit_transform(X_met)

        n_pca = min(pca_dim, n_samples - 1, X_img.shape[1])
        n_cca = min(n_pca, n_met, n_samples - 1)

        if n_cca < 1:
            print(f"Skipping {day}: insufficient dimensions")
            continue

        pca = PCA(n_components=n_pca, random_state=1)
        img_pca = pca.fit_transform(img_scaled)

        cca = CCA(n_components=n_cca, max_iter=1000)
        img_c, met_c = cca.fit_transform(img_pca, met_scaled)

        canonical_corrs = []
        for i in range(n_cca):
            corr = np.corrcoef(img_c[:, i], met_c[:, i])[0, 1]
            canonical_corrs.append(corr)

        result = {
            "day": day,
            "n_samples": n_samples,
            "n_metabolites": n_met,
            "image_pca_dim": n_pca,
            "n_cca_components": n_cca,
            "image_pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "cca1": float(canonical_corrs[0]),
            "mean_cca": float(np.mean(canonical_corrs)),
            "max_cca": float(np.max(canonical_corrs)),
        }

        for i, corr in enumerate(canonical_corrs):
            result[f"cca_{i + 1}"] = float(corr)

        results.append(result)

    cca_df = pd.DataFrame(results)
    cca_df.to_csv(output_dir / "cca_by_day.csv", index=False)

    return cca_df


def compute_metabolite_pc_correlations(aligned, output_dir, pca_dim):
    """Compute pairwise correlations between each metabolite and each image PC per day.

    Args:
        aligned: DataFrame with both image embedding and metabolite columns.
        output_dir: Path to save metabolite_pc_correlations.csv.
        pca_dim: Number of PCA components to reduce image embeddings to.

    Returns:
        DataFrame with columns day, metabolite, image_pc, correlation, abs_correlation.
    """
    rows = []
    img_cols = [c for c in aligned.columns if c.startswith("img_emb_")]

    for day in DAY_ORDER:
        day_df = aligned[aligned["day"] == day].copy()

        if len(day_df) < 10:
            continue

        met_cols = get_metabolite_columns(day)
        day_df = day_df.dropna(subset=met_cols)

        if len(day_df) < 10:
            continue

        X_img = day_df[img_cols].values
        X_img = StandardScaler().fit_transform(X_img)

        n_pca = min(pca_dim, len(day_df) - 1, X_img.shape[1])
        if n_pca < 1:
            continue

        pca = PCA(n_components=n_pca, random_state=1)
        img_pca = pca.fit_transform(X_img)

        for met in met_cols:
            met_vals = day_df[met].values.astype(float)

            for pc_idx in range(n_pca):
                corr = np.corrcoef(met_vals, img_pca[:, pc_idx])[0, 1]

                rows.append({
                    "day": day,
                    "metabolite": met,
                    "image_pc": f"PC{pc_idx + 1}",
                    "correlation": float(corr),
                    "abs_correlation": float(abs(corr)),
                })

    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(output_dir / "metabolite_pc_correlations.csv", index=False)

    return corr_df


def plot_cca(cca_df, output_dir):
    """Plot first and mean canonical correlations across days and save to disk.

    Args:
        cca_df: DataFrame with CCA results from compute_cca_by_day.
        output_dir: Path to save cca_by_day.png.
    """
    if cca_df.empty:
        return

    x = [day_to_int(d) for d in cca_df["day"]]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(x, cca_df["cca1"], "o-", label="First canonical correlation")
    ax.plot(x, cca_df["mean_cca"], "s--", label="Mean canonical correlation")

    ax.axhline(0.3, linestyle="--", alpha=0.4)
    ax.axhline(0.5, linestyle="--", alpha=0.4)
    ax.axhline(0.7, linestyle="--", alpha=0.4)

    ax.set_xlabel("Day")
    ax.set_ylabel("Canonical correlation")
    ax.set_title("Image Embeddings vs Metabolite Features: CCA by Day")
    ax.set_xticks(x)
    ax.set_xticklabels(cca_df["day"], rotation=45)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = output_dir / "cca_by_day.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved {out}")


def plot_metabolite_pc_heatmap(corr_df, output_dir):
    """Plot a heatmap of max absolute correlation between each metabolite and image PCs.

    Args:
        corr_df: DataFrame from compute_metabolite_pc_correlations.
        output_dir: Path to save metabolite_pc_heatmap.png.
    """
    if corr_df.empty:
        return

    top = (
        corr_df.sort_values("abs_correlation", ascending=False)
        .groupby(["day", "metabolite"])
        .head(1)
        .copy()
    )

    pivot = top.pivot_table(
        index="metabolite",
        columns="day",
        values="abs_correlation",
        aggfunc="max",
    )

    pivot = pivot[[d for d in DAY_ORDER if d in pivot.columns]]

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.imshow(pivot.values, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45)

    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Max Absolute Correlation Between Each Metabolite and Image PCs")
    fig.colorbar(im, ax=ax, label="max |correlation|")

    plt.tight_layout()
    out = output_dir / "metabolite_pc_heatmap.png"
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved {out}")


def main():
    """Parse arguments, load data, compute correlations, and save results and plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["rgb", "overlay"], default="rgb")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pca-dim", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(PROJ_ROOT / "analysis/combined_model/outputs/feature_correlation"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("Feature Correlation Analysis")
    print(f"Input mode: {args.input_mode}")
    print(f"Device: {device}")
    print(f"PCA dim: {args.pca_dim}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    df = load_all_data(args.input_mode)
    df.to_csv(output_dir / "loaded_records.csv", index=False)

    emb_df = extract_image_embeddings(
        df,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        device=device,
    )

    emb_df.to_csv(output_dir / "image_embeddings.csv", index=False)
    print(f"Saved image embeddings: {output_dir / 'image_embeddings.csv'}")

    aligned = prepare_aligned_features(df, emb_df)
    aligned.to_csv(output_dir / "aligned_features.csv", index=False)
    print(f"Saved aligned features: {output_dir / 'aligned_features.csv'}")

    cca_df = compute_cca_by_day(
        aligned,
        output_dir=output_dir,
        pca_dim=args.pca_dim,
    )

    corr_df = compute_metabolite_pc_correlations(
        aligned,
        output_dir=output_dir,
        pca_dim=args.pca_dim,
    )

    plot_cca(cca_df, output_dir)
    plot_metabolite_pc_heatmap(corr_df, output_dir)

    print("\nDone.")
    print("Main results:")
    print(f"  {output_dir / 'cca_by_day.csv'}")
    print(f"  {output_dir / 'metabolite_pc_correlations.csv'}")
    print(f"  {output_dir / 'cca_by_day.png'}")
    print(f"  {output_dir / 'metabolite_pc_heatmap.png'}")


if __name__ == "__main__":
    main()
