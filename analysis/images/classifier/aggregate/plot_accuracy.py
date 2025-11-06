#!/usr/bin/env python3
# Render accuracy plot with same style as Promega reference visual
# Uses exact x-axis alignment (DyXX), same line/marker formatting.

import csv, argparse, numpy as np
import matplotlib.pyplot as plt

# Your specified palette
PALETTE = [
    "#E69F00",  
    "#56B4E9",
    "#009E73",  
    "#CC79A7"  
]

def read_matrix(path):
    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]                 # ['Day', 'Model1', 'Model2', ...]
    days   = [r[0] for r in rows[1:] if r and r[0].startswith("Dy")]
    series = {}
    for j, name in enumerate(header[1:], start=1):
        vals = []
        for r in rows[1:]:
            try:
                vals.append(float(r[j]))
            except Exception:
                vals.append(np.nan)
        series[name] = vals
    return days, series

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="accuracy_by_day.csv")
    ap.add_argument("--out", default="accuracy_plot.png")
    args = ap.parse_args()

    days, series = read_matrix(args.csv)
    all_days = [int(d[2:]) for d in days]  # convert Dy03 → 3 for real x positions

    plt.figure(figsize=(12, 8))
    ax = plt.gca()
    ax.grid(True, linestyle=":", linewidth=0.8, color="0.85")

    for i, (name, ys) in enumerate(series.items()):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            all_days, ys,
            lw=2.4, ls="-", marker="o", markersize=6,
            markerfacecolor="white", markeredgewidth=1.2,
            color=color, label=name
        )

    ax.set_xticks(all_days)
    ax.set_xticklabels(days)
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Predicting Dy30 Acceptability from Earlier-Day Features", fontsize=14)
    ax.set_ylim(0.4, 1.0)
    ax.set_yticks(np.linspace(0.4, 1.0, 7))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9,
              facecolor="white", edgecolor="0.9", fontsize=10)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
