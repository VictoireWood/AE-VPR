# Command Examples

Replace all `/path/to/...` values with your local dataset and checkpoint paths.

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

## QD01 Evaluation

```bash
python eval_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/qd01_test_root \
  --dataset_name qd01 \
  --test_set_list qd01 \
  --resume_model /path/to/best_model.pth \
  --classifier QAMC
```

## QD02 Evaluation

```bash
python eval_dc.py \
  --train_set_path /path/to/train_root \
  --test_set_path /path/to/qd02_test_root \
  --dataset_name qd02 \
  --test_set_list qd02 \
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
