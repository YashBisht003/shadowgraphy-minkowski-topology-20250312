#!/usr/bin/env python
"""
Estimate image scale (mm/px) from a scale .cine containing a horizontal cylinder.

Assumption:
- The cylinder true diameter is known (default 1/4 in = 6.35 mm).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pycine.file import read_header
from pycine.raw import read_frames


def load_mean_frame(cine_path: str, max_frames: int = 64) -> np.ndarray:
    header = read_header(cine_path)
    n = int(header["cinefileheader"].ImageCount)
    use_n = min(n, max_frames) if n > 0 else 1
    frames = []
    for frame in read_frames(cine_path, start_frame=1, count=use_n)[0]:
        frames.append(frame.astype(np.float32))
    if not frames:
        raise RuntimeError("No frames read from cine.")
    return np.mean(np.stack(frames, axis=0), axis=0)


def detect_diameter_pixels(img_norm: np.ndarray) -> Dict[str, np.ndarray | float]:
    img = cv2.GaussianBlur(img_norm.astype(np.float32), (0, 0), 1.2)
    h, w = img.shape

    samples = []
    for x in range(0, w, 2):
        col = img[:, x]
        grad = np.gradient(col)

        top_lo, top_hi = int(0.15 * h), int(0.65 * h)
        bot_lo, bot_hi = int(0.35 * h), int(0.90 * h)

        y_top = int(np.argmin(grad[top_lo:top_hi]) + top_lo)
        y_bot = int(np.argmax(grad[bot_lo:bot_hi]) + bot_lo)

        diam = float(y_bot - y_top)
        quality = float(abs(grad[y_top]) + abs(grad[y_bot]))
        samples.append((x, y_top, y_bot, diam, quality))

    arr = np.asarray(samples, dtype=float)

    valid = (arr[:, 3] > 0.12 * h) & (arr[:, 3] < 0.80 * h)
    if np.count_nonzero(valid) < 20:
        raise RuntimeError("Unable to find enough valid edge samples.")

    q_thr = np.percentile(arr[valid, 4], 65)
    valid &= arr[:, 4] >= q_thr
    cand = arr[valid]
    if len(cand) < 20:
        raise RuntimeError("Edge quality filter removed too many samples.")

    diam_med = float(np.median(cand[:, 3]))
    stable = np.abs(cand[:, 3] - diam_med) <= 5.0
    final = cand[stable]
    if len(final) < 20:
        final = cand

    return {
        "all_samples": arr,
        "final_samples": final,
        "diameter_px_median": float(np.median(final[:, 3])),
        "diameter_px_mean": float(np.mean(final[:, 3])),
        "diameter_px_std": float(np.std(final[:, 3])),
        "x_min": float(np.min(final[:, 0])),
        "x_max": float(np.max(final[:, 0])),
    }


def save_diagnostic_plot(
    img_norm: np.ndarray, final_samples: np.ndarray, diameter_px: float, out_png: Path
) -> None:
    xs = final_samples[:, 0]
    y_top = final_samples[:, 1]
    y_bot = final_samples[:, 2]
    diam = final_samples[:, 3]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=130)
    axes[0].imshow(img_norm, cmap="gray")
    axes[0].plot(xs, y_top, "r.", ms=1.5, label="Top edge")
    axes[0].plot(xs, y_bot, "c.", ms=1.5, label="Bottom edge")
    axes[0].set_title("Detected Cylinder Edges")
    axes[0].set_xlim([max(0, np.min(xs) - 80), min(img_norm.shape[1], np.max(xs) + 80)])
    axes[0].set_ylim([img_norm.shape[0], 0])
    axes[0].legend(loc="upper right")

    axes[1].plot(xs, diam, lw=1.0)
    axes[1].axhline(diameter_px, color="r", ls="--", label=f"median={diameter_px:.2f}px")
    axes[1].set_title("Diameter vs X")
    axes[1].set_xlabel("x (px)")
    axes[1].set_ylabel("diameter (px)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate mm/px from scale .cine")
    parser.add_argument("--cine", required=True, type=Path, help="Path to scale cine")
    parser.add_argument("--diameter-mm", type=float, default=6.35, help="Known cylinder diameter in mm")
    parser.add_argument("--out-dir", type=Path, default=Path("shadowgraphy_topology/scale_calibration"))
    args = parser.parse_args()

    cine_path = str(args.cine)
    if not args.cine.exists():
        raise FileNotFoundError(f"File not found: {args.cine}")
    if args.diameter_mm <= 0:
        raise ValueError("--diameter-mm must be positive.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    mean_frame = load_mean_frame(cine_path)
    img_norm = (mean_frame - mean_frame.min()) / (mean_frame.max() - mean_frame.min() + 1e-9)

    det = detect_diameter_pixels(img_norm)
    d_px = det["diameter_px_median"]
    mm_per_px = args.diameter_mm / d_px
    px_per_mm = d_px / args.diameter_mm
    um_per_px = mm_per_px * 1000.0

    diag_png = args.out_dir / "scale_detection.png"
    save_diagnostic_plot(img_norm, det["final_samples"], d_px, diag_png)

    summary = {
        "cine_path": str(args.cine.resolve()),
        "known_diameter_mm": float(args.diameter_mm),
        "detected_diameter_px_median": float(d_px),
        "detected_diameter_px_mean": float(det["diameter_px_mean"]),
        "detected_diameter_px_std": float(det["diameter_px_std"]),
        "scale_mm_per_px": float(mm_per_px),
        "scale_px_per_mm": float(px_per_mm),
        "scale_um_per_px": float(um_per_px),
        "fit_x_range_px": [float(det["x_min"]), float(det["x_max"])],
        "num_edge_samples": int(len(det["final_samples"])),
        "diagnostic_plot": str(diag_png.resolve()),
    }

    summary_path = args.out_dir / "scale_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Scale calibration complete.")
    print(f"diameter_px={d_px:.3f}")
    print(f"mm_per_px={mm_per_px:.8f}")
    print(f"px_per_mm={px_per_mm:.5f}")
    print(f"um_per_px={um_per_px:.3f}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Plot: {diag_png.resolve()}")


if __name__ == "__main__":
    main()
