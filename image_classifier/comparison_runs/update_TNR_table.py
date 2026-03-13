#!/usr/bin/env python3
"""Regenerate per_day_study_TNR_table.md from results in per_day_study, per_day_study_overlay, per_day_study_rgb_mask."""

import json
from pathlib import Path

DAYS = [6, 8, 10, 13, 15, 17, 20.5, 24, 26, 28, 30]
SETUPS = [
    ("RGB", "per_day_study"),
    ("Overlay", "per_day_study_overlay"),
    ("RGB + mask", "per_day_study_rgb_mask"),
]
MODELS = ["per_day", "cnn_lstm", "effnet_ts"]
BASE = Path(__file__).parent


def day_dir(d):
    return f"day_{d}_0" if d != 20.5 else "day_20_5"


def get_tnr(setup_dir, model, day):
    p = BASE / setup_dir / model / day_dir(day) / "results.json"
    if not p.exists():
        return None
    with open(p) as f:
        r = json.load(f)
    t = r.get("test_at_optimal") or r.get("test_at_0.5")
    return round(t["TNR"], 2) if t else None


def main():
    out = [
        "# Per-day study — TNR (True Negative Rate) by setup",
        "",
        "Test set at optimal threshold. Higher TNR = fewer false positives on negatives.",
        "",
    ]
    for label, dirname in SETUPS:
        out.append("---")
        out.append("")
        out.append(f"## {label} (`{dirname}`)")
        out.append("")
        out.append("| Day | per_day | cnn_lstm | effnet_ts |")
        out.append("|-----|---------|----------|-----------|")
        for day in DAYS:
            row = [str(day)]
            for model in MODELS:
                tnr = get_tnr(dirname, model, day)
                row.append(f"{tnr:.2f}" if tnr is not None else "—")
            out.append("| " + " | ".join(row) + " |")
        out.append("")
    out.append("---")
    with open(BASE / "per_day_study_TNR_table.md", "w") as f:
        f.write("\n".join(out))
    print("Updated per_day_study_TNR_table.md")


if __name__ == "__main__":
    main()
