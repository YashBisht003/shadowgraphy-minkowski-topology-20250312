#!/usr/bin/env python
"""
Compute 2D Minkowski functionals from a Phantom .cine backlit-shadowgraphy video.

Outputs:
- per_frame_minkowski.csv
- summary.json
- timeseries.png
- phase_area_vs_euler.png
- segmentation_previews.png
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycine.file import read_header
from pycine.raw import read_frames
from skimage.filters import threshold_otsu
from skimage.measure import euler_number, perimeter
from tqdm import tqdm


@dataclass
class ProcessingConfig:
    blur_kernel: int
    right_crop_frac: float
    border_frac: float
    min_component_area: int
    stride: int


def make_odd_kernel(k: int) -> int:
    if k < 3:
        return 3
    return k if k % 2 == 1 else k + 1


def build_roi_mask(shape: Tuple[int, int], right_crop_frac: float, border_frac: float) -> np.ndarray:
    h, w = shape
    roi = np.zeros((h, w), dtype=bool)

    border = max(1, int(round(min(h, w) * border_frac)))
    x0 = border
    y0 = border
    x1 = int(round(w * (1.0 - right_crop_frac)))
    y1 = h - border

    x1 = min(max(x1, x0 + 1), w - border)
    y1 = min(max(y1, y0 + 1), h - border)
    roi[y0:y1, x0:x1] = True
    return roi


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n_labels <= 1:
        return mask

    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def segment_frame(frame_u16: np.ndarray, roi_mask: np.ndarray, blur_kernel: int, min_component_area: int) -> Tuple[np.ndarray, np.ndarray, float]:
    frame = frame_u16.astype(np.float32)
    local_bg = cv2.blur(frame, (blur_kernel, blur_kernel))
    enhanced = np.clip(local_bg - frame, 0.0, None)

    roi_values = enhanced[roi_mask]
    if roi_values.size == 0:
        raise RuntimeError("ROI mask is empty. Increase ROI area.")

    if float(roi_values.max()) == float(roi_values.min()):
        thr = float(roi_values.mean())
    else:
        thr = float(threshold_otsu(roi_values))

    mask = (enhanced > thr) & roi_mask
    mask = remove_small_components(mask, min_component_area)
    return mask, enhanced, thr


def summarize_column(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def save_timeseries_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, dpi=140)

    axes[0].plot(df["time_s"], df["area_fraction"], lw=1.0)
    axes[0].set_ylabel("Area Fraction")
    axes[0].grid(alpha=0.25)

    axes[1].plot(df["time_s"], df["perimeter_density"], lw=1.0, color="tab:orange")
    axes[1].set_ylabel("Perimeter Density")
    axes[1].grid(alpha=0.25)

    axes[2].plot(df["time_s"], df["euler_density"], lw=1.0, color="tab:green")
    axes[2].set_ylabel("Euler Density")
    axes[2].grid(alpha=0.25)

    axes[3].plot(df["time_s"], df["threshold_enhanced"], lw=1.0, color="tab:red")
    axes[3].set_ylabel("Otsu Thresh")
    axes[3].set_xlabel("Time From First Analyzed Frame (s)")
    axes[3].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_phase_plot(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
    points = ax.scatter(df["area_fraction"], df["euler_density"], c=df["time_s"], s=8, cmap="viridis")
    cbar = fig.colorbar(points, ax=ax)
    cbar.set_label("Time (s)")
    ax.set_xlabel("Area Fraction")
    ax.set_ylabel("Euler Density")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_segmentation_previews(previews: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float]], out_path: Path) -> None:
    if not previews:
        return

    rows = len(previews)
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), dpi=120)
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (title, raw, enhanced, mask, thr) in enumerate(previews):
        axes[i, 0].imshow(raw, cmap="gray")
        axes[i, 0].set_title(f"{title} raw")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(enhanced, cmap="magma")
        axes[i, 1].set_title(f"{title} enhanced")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask, cmap="gray")
        axes[i, 2].set_title(f"{title} mask (thr={thr:.2f})")
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Minkowski topology analysis for Phantom .cine data")
    parser.add_argument("--cine", required=True, type=Path, help="Path to input .cine file")
    parser.add_argument("--out-dir", type=Path, default=Path("shadowgraphy_topology_output"))
    parser.add_argument("--start-frame", type=int, default=1, help="1-based frame index")
    parser.add_argument("--count", type=int, default=None, help="Number of frames to read from start-frame")
    parser.add_argument("--stride", type=int, default=2, help="Analyze every Nth frame")
    parser.add_argument("--blur-kernel", type=int, default=91, help="Odd kernel size for local background")
    parser.add_argument("--right-crop-frac", type=float, default=0.06, help="Ignore rightmost fraction of image")
    parser.add_argument("--border-frac", type=float, default=0.01, help="Ignore border fraction on all sides")
    parser.add_argument("--min-component-area", type=int, default=9, help="Drop connected components smaller than this")
    args = parser.parse_args()

    if not args.cine.exists():
        raise FileNotFoundError(f"CINE file not found: {args.cine}")
    if args.start_frame < 1:
        raise ValueError("--start-frame must be >= 1")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    config = ProcessingConfig(
        blur_kernel=make_odd_kernel(args.blur_kernel),
        right_crop_frac=float(args.right_crop_frac),
        border_frac=float(args.border_frac),
        min_component_area=int(args.min_component_area),
        stride=int(args.stride),
    )

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    header = read_header(str(args.cine))
    cfh = header["cinefileheader"]
    bih = header["bitmapinfoheader"]
    setup = header["setup"]

    total_frames = int(cfh.ImageCount)
    first_image_no = int(cfh.FirstImageNo)
    fps = float(setup.FrameRate) if float(setup.FrameRate) > 0 else 1.0
    width = int(bih.biWidth)
    height = int(bih.biHeight)

    remaining = total_frames - args.start_frame + 1
    if remaining <= 0:
        raise ValueError(f"start-frame {args.start_frame} exceeds frame count {total_frames}")

    count = remaining if args.count is None else min(int(args.count), remaining)
    roi_mask = build_roi_mask((height, width), config.right_crop_frac, config.border_frac)
    roi_pixels = int(np.count_nonzero(roi_mask))
    if roi_pixels == 0:
        raise RuntimeError("ROI contains zero pixels. Adjust ROI parameters.")

    frame_gen, _, _ = read_frames(str(args.cine), start_frame=args.start_frame, count=count)

    records: List[Dict[str, float]] = []
    preview_first = None
    preview_mid = None
    preview_last = None
    mid_target = args.start_frame + (count // 2)

    for i, frame in enumerate(tqdm(frame_gen, total=count, desc="Analyzing frames"), start=0):
        cine_index = args.start_frame + i
        if i % config.stride != 0:
            continue

        mask, enhanced, thr = segment_frame(frame, roi_mask, config.blur_kernel, config.min_component_area)
        area_px = int(np.count_nonzero(mask))
        perim_px = float(perimeter(mask, neighborhood=8))
        chi = int(euler_number(mask, connectivity=2))

        analyzed_frame_number = first_image_no + cine_index - 1
        t = (cine_index - args.start_frame) / fps
        t_trigger = analyzed_frame_number / fps

        records.append(
            {
                "cine_index_1based": cine_index,
                "cine_frame_number": analyzed_frame_number,
                "time_s": t,
                "time_from_trigger_s": t_trigger,
                "threshold_enhanced": thr,
                "area_px": area_px,
                "area_fraction": area_px / roi_pixels,
                "perimeter_px": perim_px,
                "perimeter_density": perim_px / roi_pixels,
                "euler_chi": chi,
                "euler_density": chi / roi_pixels,
            }
        )

        if preview_first is None:
            preview_first = (f"Frame {cine_index}", frame, enhanced, mask, thr)
        if preview_mid is None and cine_index >= mid_target:
            preview_mid = (f"Frame {cine_index}", frame, enhanced, mask, thr)
        preview_last = (f"Frame {cine_index}", frame, enhanced, mask, thr)

    if not records:
        raise RuntimeError("No frames analyzed. Check --count/--stride settings.")

    df = pd.DataFrame.from_records(records)
    csv_path = out_dir / "per_frame_minkowski.csv"
    df.to_csv(csv_path, index=False)

    timeseries_path = out_dir / "timeseries.png"
    phase_path = out_dir / "phase_area_vs_euler.png"
    previews_path = out_dir / "segmentation_previews.png"
    save_timeseries_plot(df, timeseries_path)
    save_phase_plot(df, phase_path)

    previews: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, float]] = []
    if preview_first is not None:
        previews.append(preview_first)
    if preview_mid is not None:
        previews.append(preview_mid)
    if preview_last is not None:
        previews.append(preview_last)
    save_segmentation_previews(previews, previews_path)

    metrics_summary = {}
    for col in ["area_fraction", "perimeter_density", "euler_chi", "euler_density", "threshold_enhanced"]:
        metrics_summary[col] = summarize_column(df[col].to_numpy())

    trend_slopes = {}
    if len(df) >= 2 and not math.isclose(float(df["time_s"].iloc[-1]), float(df["time_s"].iloc[0])):
        for col in ["area_fraction", "perimeter_density", "euler_density"]:
            slope = np.polyfit(df["time_s"].to_numpy(), df[col].to_numpy(), deg=1)[0]
            trend_slopes[f"{col}_slope_per_s"] = float(slope)

    extrema = {
        "max_area_fraction_frame": int(df.loc[df["area_fraction"].idxmax(), "cine_index_1based"]),
        "min_area_fraction_frame": int(df.loc[df["area_fraction"].idxmin(), "cine_index_1based"]),
        "max_euler_density_frame": int(df.loc[df["euler_density"].idxmax(), "cine_index_1based"]),
        "min_euler_density_frame": int(df.loc[df["euler_density"].idxmin(), "cine_index_1based"]),
    }

    corr = {
        "area_vs_perimeter_density": float(df["area_fraction"].corr(df["perimeter_density"])),
        "area_vs_euler_density": float(df["area_fraction"].corr(df["euler_density"])),
        "perimeter_vs_euler_density": float(df["perimeter_density"].corr(df["euler_density"])),
    }

    summary = {
        "cine_path": str(args.cine.resolve()),
        "image_shape_hw": [height, width],
        "fps": fps,
        "first_image_no": first_image_no,
        "total_frames_in_cine": total_frames,
        "analyzed_start_frame": args.start_frame,
        "analyzed_count_read": count,
        "analyzed_stride": config.stride,
        "analyzed_frames_effective": int(len(df)),
        "roi_pixels": roi_pixels,
        "roi_fraction_of_frame": float(roi_pixels / (height * width)),
        "config": {
            "blur_kernel": config.blur_kernel,
            "right_crop_frac": config.right_crop_frac,
            "border_frac": config.border_frac,
            "min_component_area": config.min_component_area,
        },
        "metrics_summary": metrics_summary,
        "trend_slopes": trend_slopes,
        "correlations": corr,
        "extrema": extrema,
        "outputs": {
            "csv": str(csv_path.resolve()),
            "timeseries_plot": str(timeseries_path.resolve()),
            "phase_plot": str(phase_path.resolve()),
            "preview_plot": str(previews_path.resolve()),
        },
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Analyzed {len(df)} frames from {args.cine}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Summary: {summary_path.resolve()}")
    print(f"Plots: {timeseries_path.resolve()}, {phase_path.resolve()}, {previews_path.resolve()}")


if __name__ == "__main__":
    main()
