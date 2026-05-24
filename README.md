# AE-VPR

Official code release for **Altitude-Adaptive Vision-Only Geo-Localization for UAVs in GPS-Denied Environments**.

AE-VPR addresses scale mismatch in UAV visual place recognition when the flight altitude changes. The pipeline first estimates relative altitude from a single nadir image, then uses the estimate to crop the query image to a canonical map scale before visual place recognition.

## Highlights

- **Relative altitude estimation (RAE)**: transforms RGB nadir images into frequency-domain representations with 2D FFT and predicts altitude intervals through regression-as-classification.
- **Altitude-adaptive VPR**: uses the estimated altitude to normalize the query scale before classification-then-retrieval localization.
- **QAMC classifier**: introduces a quality-adaptive margin classifier for aerial images with different sharpness and quality.
- **WCE refinement**: provides weighted coordinate estimation over retrieved candidates for sub-grid coordinate recovery.
- **No range sensor required at inference**: the online pipeline uses visual input only.

## Method Overview

The paper decomposes the system into two stages:

1. **Relative altitude estimation**
   `Spat2Freq -> altitude classifier -> Class2Alt`

2. **Altitude-adaptive localization**
   `Crop(query, estimated altitude) -> classify candidate cells -> retrieve references -> optional WCE`

The default code settings follow the main RAE configuration in the paper: `ResNet50`, `MixVPR`, `QAMC`, FFT preprocessing enabled, log base `1.5`, input size `336 x 448`, and altitude interval width `M=50`.

## Datasets

This release is scoped to the datasets used in the paper: two synthetic datasets and two real-flight datasets.

| Dataset | Region | Acquisition | Test images | Search area | Altitude range |
| --- | --- | --- | ---: | --- | --- |
| CT01 | Beijing | Cropped satellite map with degradation | 500 | 7.8 x 5.9 km | 100-700 m |
| CT02 | Shanghai | Cropped satellite map with degradation | 500 | 8.8 x 4.9 km | 100-700 m |
| QD01 | Qingdao | UAV real-flight | 814 | 4.8 x 3.5 km | 100-650 m |
| QD02 | Qingdao | UAV real-flight | 470 | 4.8 x 3.5 km | 100-650 m |

Download links:

- Baidu Netdisk: [https://pan.baidu.com/s/1PyeSPynSudF6BqIkumJv1w](https://pan.baidu.com/s/1PyeSPynSudF6BqIkumJv1w), extraction code: `pjyq`
- OneDrive: [https://1drv.ms/u/c/159b6aa0963a5434/IQAlnZjd6irCRrAd2yYyfFjPASzz5SIVKc4aQkPT9eawtJ4](https://1drv.ms/u/c/159b6aa0963a5434/IQAlnZjd6irCRrAd2yYyfFjPASzz5SIVKc4aQkPT9eawtJ4)

Datasets, checkpoints, logs, and generated tiles are intentionally excluded from git.

## Repository Layout

```text
.
├── train_dc.py                 # Train the grouped altitude classifier
├── eval_dc.py                  # Evaluate a checkpoint on test sets
├── test_dc.py                  # Evaluate and export per-image prediction CSV
├── eval_time_performance.py    # Measure latency and memory
├── parser.py                   # Shared command-line arguments
├── dataloaders/                # Dataset definitions and filename parsers
├── models/                     # Backbones, aggregators, and classifiers
├── utils/                      # Inference, FFT, retrieval, WCE, and checkpoint helpers
├── docs/DATA_FORMAT.md         # Dataset layout details
└── examples/commands.md        # Common command examples
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install a CUDA-compatible PyTorch build for your system before or while installing the requirements.

`utils/retrieval.py` can use FAISS when installed, and otherwise falls back to a NumPy backend for small reference sets.

## Data Format

Training and test roots are passed through command-line arguments. The main expected layout is:

```text
DATA_ROOT/
├── Dataframes/
│   └── <dataset_name>.csv
└── Images/
    └── <dataset_name>/
        └── @<year>@<flight_height>@<rotation_angle>@<loc_x>@<loc_y>@.png
```

The CSV should include:

```text
year, origin_img, flight_height, flight_class, rotation_angle, loc_x, loc_y
```

More details are in [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md).

## Training

Example CT01 training command:

```bash
python train_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct01 \
  --test_set_list ct01 \
  --classifier QAMC \
  --backbone resnet50 \
  --layers_to_crop 4 \
  --aggregator MixVPR \
  --train_resize 336 448 \
  --test_resize 336 \
  --M 50 \
  --N 1 \
  --batch_size 64 \
  --epochs_num 500
```

For the spatial-domain ablation, disable FFT preprocessing:

```bash
python train_dc.py ... --no-fft
```

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct01 \
  --test_set_list ct01 \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```

Export per-image predictions:

```bash
python test_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct01 \
  --test_set_list ct01 \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```

Measure runtime:

```bash
python eval_time_performance.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct01 \
  --test_set_list ct01 \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```

## Notes

- Large files are ignored by `.gitignore`, including `data/`, `datasets/`, `logs/`, `cache/`, checkpoints, NumPy arrays, and model weights.
- The code assumes dataset-specific filename metadata for altitude and coordinate parsing. Keep the downloaded dataset layout unless you also update the dataloaders.
- Public dataset names are limited to the four datasets evaluated in the paper: `ct01`, `ct02`, `qd01`, and `qd02`.

## Citation

If this repository is useful for your research, please cite the paper:

```bibtex
@misc{shao2026altitude,
  title         = {Altitude-Adaptive Vision-Only Geo-Localization for UAVs in GPS-Denied Environments},
  author        = {Shao, Xingyu and He, Mengfan and Sun, Liangzheng and Li, Chunyu and Meng, Ziyang},
  year          = {2026},
  eprint        = {2602.23872},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
