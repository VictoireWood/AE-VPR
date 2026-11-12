# Altitude-Adaptive UAV-to-Map Image Matching for Aerial Geo-Localization

Altitude estimation, altitude-guided query cropping, and FAMC place recognition.

## Installation

Python 3.10+, PyTorch 2.3+, and torchvision 0.18+.

```bash
pip install -r requirements.txt
```

## Data

Download [release_data.zip](https://pan.baidu.com/s/1BkaELLkf6qgfYrB-U-h4dQ?pwd=3vbh), extraction code `3vbh`. Extract it and point `--data` to the resulting `release_data` directory:

```text
/path/to/data/
├── Map/
│   ├── CT01/             # 201810.tif, …, 202306.tif
│   ├── CT02/
│   └── QD/
└── Query/
    ├── CT01/            # 500 images
    ├── CT02/            # 500 images
    ├── QD01/            # 814 images
    └── QD02/            # 470 images
```

Maps are named `YYYYMM.tif`. Altitude training uses 202306. VPR uses CT01's five and CT02's six dates other than 202306, and all five QD dates.

Query and crop filenames use `@angle@altitude@easting@northing@.png` or `.jpg`. Angles are in degrees; altitudes and coordinates are in metres. Queries, training crops, galleries, and evaluation use one fixed WGS84 / UTM zone per region: CT01 uses 50N (EPSG:32650); CT02 and QD use 51N (EPSG:32651).

## Cropping

Run these commands for each region, selecting `CT01`, `CT02`, or `QD`:

```bash
python prepare_crops.py --data /path/to/data --region QD --mode vpr
python prepare_crops.py --data /path/to/data --region QD --mode gallery
python prepare_crops.py --data /path/to/data --region QD --mode height
```

Output: `Crops/REGION/{height,vpr}/{train,gallery}/DATE/`.

VPR centres are spaced 42.6666667 m east–west and 32 m north–south along the map axes, with a margin for rotated 200 m views. Gallery crops share the training centres. `--spacing-m EAST NORTH` changes the spacing; `--bounds E_MIN N_MIN E_MAX N_MAX` selects an area.

Altitude crops cover 100–695 m in 5 m steps. `--heights-m` selects altitudes; `--angles-degrees` selects crop angles. VPR crops use 150 m with probability 0.5, otherwise a height sampled uniformly from [100,200) m; angles are sampled from 0, 30, …, 330 degrees. Gallery crops use 150 m and 0 degrees. `--seed` controls sampling.

The camera geometry is 2048×1536 pixels with focal length 1200 pixels. VPR/gallery crops are 224×224; altitude crops are 480×360.

## Training

```bash
python train.py --task height --data /path/to/data --region QD \
  --backbone-checkpoint /path/to/resnet50_imagenet_v1.pth \
  --output-dir outputs/height --device cuda:0

python train.py --task vpr --data /path/to/data --region QD \
  --backbone-checkpoint /path/to/efficientnet_b5_imagenet.pth \
  --output-dir outputs/vpr --device cuda:0
```

| Setting | Altitude estimation | VPR |
| --- | --- | --- |
| Input H×W | 336×448 | 224×224 |
| Backbone | ResNet50 through layer3 | EfficientNet-B5 features |
| Aggregator | MixVPR, 1024 dimensions | MixVPR, 3072 dimensions |
| Frozen parameters | conv1, bn1, layer1 | First 5 feature blocks |
| Classifier | AAMC, s=100, m=0.2 | FAMC, s=30, m=0.4 |
| Batch size | 64 | 32 |

Adam learning rates are `--model-lr 1e-4` and `--head-lr 1e-2`. Spat2Freq is enabled for altitude estimation; `--no-frequency` selects RGB. Classes cover [100,700) m in 50 m intervals, with centres 125, 175, …, 675 m.

Backbone initialization uses torchvision ImageNet state dictionaries. CUDA uses mixed precision; `--no-amp` selects FP32. Checkpoints are saved to `checkpoint.pth` and `epochs/epoch_NNN/checkpoint.pth`. Resume with the original training options and `--resume /path/to/checkpoint.pth`.

## Evaluation

```bash
python evaluate.py --task height --data /path/to/data --dataset QD01 \
  --checkpoint /path/to/checkpoint.pth \
  --output-dir results/height --device cuda:0

python evaluate.py --task vpr --data /path/to/data --dataset QD01 \
  --checkpoint /path/to/checkpoint.pth \
  --height-checkpoint /path/to/height/checkpoint.pth \
  --renormalize-descriptor --output-dir results/vpr --device cuda:0
```

`--dataset` selects CT01, CT02, QD01, or QD02. QD01 and QD02 share `Crops/QD/vpr/gallery`. `--query-dir` and `--gallery-dir` select directories directly.

`--crop-mode predicted` crops the query by `min(1,150/predicted_altitude)` before resizing to 224×224. `gt` uses the labelled altitude; `none` uses the full image. Use `--renormalize-descriptor` for QD.

Scores use the maximum inner product across dates at each centre. Metrics are altitude MAE, P(error < 25/50/100 m), and R@1/R@5 at distances ≤25/50/100 m. Distances are computed directly from query and gallery UTM coordinates.

Outputs: `metrics.json` and `predictions.json`.

## Single-Image Inference

```bash
python demo.py --query-image /path/to/query.png \
  --gallery-dir /path/to/data/Crops/QD/vpr/gallery \
  --height-checkpoint /path/to/height/checkpoint.pth \
  --retrieval-checkpoint /path/to/vpr/checkpoint.pth \
  --renormalize-descriptor --device cuda:0 --output results/query.json
```

Returns the predicted altitude, crop box, ranked places, and scores. Each entry point supports `--help`.

## License

Original code: [AGPL-3.0-only](LICENSE). MixVPR follows the [upstream implementation](https://github.com/amaralibey/MixVPR); AAMC references [ArcFace](https://github.com/ronghuaiyang/arcface-pytorch). Third-party code, weights, and data retain their respective licenses.
