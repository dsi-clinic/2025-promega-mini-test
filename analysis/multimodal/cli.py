#!/usr/bin/env python3
"""Argparse + config dict for the multimodal trainer."""

import argparse

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multimodal Organoid Classification")

    parser.add_argument("--backbone", choices=["vit", "resnet", "efficientnet"], default="vit")
    parser.add_argument("--input-mode", choices=["rgb", "overlay", "rgb_mask", "overlay_mask"], default="rgb")
    parser.add_argument("--fusion-strategy", choices=["concat", "gated"], default="concat",
                        help="Fusion strategy: concat or gated (metabolite modulates image)")

    parser.add_argument("--use-images", action="store_true", default=True)
    parser.add_argument("--use-metabolites", action="store_true", default=False)
    parser.add_argument("--images-only", action="store_true", help="Use only images (no metabolites)")
    parser.add_argument("--metabolites-only", action="store_true", help="Use only metabolites (no images)")

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-epochs-phase1", type=int, default=50)
    parser.add_argument("--num-epochs-phase2", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--use-augmentation", action="store_true")

    parser.add_argument("--days", nargs="*", default=None,
                        help="Specific days to train (e.g., Dy03 Dy06)")

    parser.add_argument("--all-data", default="data/all_data.json",
                        help="Path to all_data.json (single source of truth)")
    parser.add_argument("--splits-csv", default="data/splits/canonical_2026_winter.csv",
                        help="Organoid-level train/val/test split CSV (loaded via Splits.from_csv)")
    parser.add_argument("--mode", default="base",
                        choices=["base", "switch1", "switch2", "switch3"],
                        help="Filter preset (see pipeline.data_loader.filters_for_mode)")
    parser.add_argument("--output-dir", default="analysis/multimodal/outputs_multimodal")

    args = parser.parse_args()

    if args.images_only:
        args.use_images = True
        args.use_metabolites = False
    elif args.metabolites_only:
        args.use_images = False
        args.use_metabolites = True

    return args


def build_config(args: argparse.Namespace) -> dict:
    """Translate parsed args into the config dict consumed by data/models/train modules."""
    return {
        "backbone": args.backbone,
        "input_mode": args.input_mode,
        "fusion_strategy": args.fusion_strategy,
        "use_images": args.use_images,
        "use_metabolites": args.use_metabolites,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs_phase1": args.num_epochs_phase1,
        "num_epochs_phase2": args.num_epochs_phase2,
        "early_stopping_patience": args.early_stopping_patience,
        "target_size": (384, 512),
        "use_augmentation": args.use_augmentation,
        "all_data_path": args.all_data,
        "splits_csv": args.splits_csv,
        "mode": args.mode,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def print_config(config: dict) -> None:
    print("\n" + "=" * 70)
    print("MULTIMODAL EXPERIMENT CONFIGURATION")
    print("=" * 70)
    for k, v in config.items():
        print(f"{k:30s}: {v}")
    print("=" * 70 + "\n")
