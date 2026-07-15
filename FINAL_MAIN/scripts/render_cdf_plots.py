"""Render CDF plots from evaluation CSVs.

Reads raw_pos_cdf.csv, full_pos_cdf.csv, and full_latency_cdf.csv and writes
cdf_position.png and cdf_latency.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv

import matplotlib.pyplot as plt


def read_cdf(path: Path) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["value"]))
            ys.append(float(row["cdf"]))
    return xs, ys


def main() -> None:
    parser = argparse.ArgumentParser(description="Render CDF plots from CSV files")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Directory containing CSV files")
    parser.add_argument("--out-dir", type=Path, default=Path("."), help="Directory to write PNG files")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_pos = data_dir / "raw_pos_cdf.csv"
    full_pos = data_dir / "full_pos_cdf.csv"
    full_lat = data_dir / "full_latency_cdf.csv"

    x_raw, y_raw = read_cdf(raw_pos)
    x_full, y_full = read_cdf(full_pos)

    plt.figure(figsize=(5.2, 3.4))
    plt.plot(x_raw, y_raw, label="Raw", color="#16324F", linewidth=2)
    plt.plot(x_full, y_full, label="Full", color="#0F766E", linewidth=2)
    plt.xlabel("Position error (mm)")
    plt.ylabel("CDF")
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_position.png", dpi=200)
    plt.close()

    x_lat, y_lat = read_cdf(full_lat)
    plt.figure(figsize=(5.2, 3.4))
    plt.plot(x_lat, y_lat, color="#334155", linewidth=2)
    plt.xlabel("Latency (ms)")
    plt.ylabel("CDF")
    plt.xlim(0, 2500)
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cdf_latency.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
