# Shadowgraphy Minkowski Topology Analysis

This repository contains:

- `analyze_minkowski_cine.py`: Python pipeline for 2D Minkowski-functional topology analysis of Phantom `.cine` backlit-shadowgraphy frames.
- `results_20250312_bg1_ql0p2_qg34/`: generated outputs for:
  - `D:\backlit shadowgraphy\20250312_bg=1_Ql=0.2_mLPM_Qg=34_SLPM_.cine`

## Output Files

- `per_frame_minkowski.csv`: per-frame metrics (`area`, `perimeter`, `Euler chi`, normalized densities, threshold).
- `summary.json`: aggregate statistics, correlations, extrema, and processing config.
- `timeseries.png`: time traces of area fraction, perimeter density, Euler density, and threshold.
- `phase_area_vs_euler.png`: area-fraction vs Euler-density phase cloud.
- `segmentation_previews.png`: representative frame segmentation diagnostics.

## Environment

Use Python 3.11+.

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

```bash
python analyze_minkowski_cine.py \
  --cine "D:\backlit shadowgraphy\20250312_bg=1_Ql=0.2_mLPM_Qg=34_SLPM_.cine" \
  --out-dir results_20250312_bg1_ql0p2_qg34 \
  --stride 2
```

## Notes

- This workflow uses local background subtraction and Otsu thresholding within a fixed ROI.
- Minkowski functionals are computed on a binary spray mask:
  - Area (pixel count)
  - Perimeter (`skimage.measure.perimeter`)
  - Euler characteristic (`skimage.measure.euler_number`)
