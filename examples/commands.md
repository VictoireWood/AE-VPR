# Command Examples

Replace all `/path/to/...` values with your local dataset and checkpoint paths.

## Debug Dataset

```bash
python tools/make_debug_dataset.py --overwrite
```

Run a minimal CPU smoke test on the generated data:

```bash
python train_dc.py \
  --train_set_path data/debug/height_classification/train \
  --test_set_path data/debug/height_classification/test \
  --dataset_name debug_height \
  --test_set_list debug_height \
  --device cpu \
  --epochs_num 1 \
  --iterations_per_epoch 2 \
  --batch_size 2 \
  --min_images_per_class 1
```

## CT01 Training

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

## CT02 Training

```bash
python train_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct02 \
  --test_set_list ct02 \
  --classifier QAMC \
  --backbone resnet50 \
  --layers_to_crop 4 \
  --aggregator MixVPR \
  --M 50 \
  --N 1
```

## Real-Flight Evaluation

```bash
python test_dc.py \
  --train_set_path /path/to/train_root \
  --val_set_path /path/to/real_photo_root \
  --dataset_name 2022 \
  --test_set_list real_photo \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```

## Runtime Evaluation

```bash
python eval_time_performance.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/test_root \
  --dataset_name ct01 \
  --test_set_list ct01 \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```
