import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import logging
from datetime import datetime
import sys
import torchmetrics
from tqdm import tqdm
from math import sqrt
import numpy as np
import platform

from dataloaders.HCDataset import realHCDataset_N, InfiniteDataLoader, HCDataset_shN, TestDataset, TestDatasetNew, visloc_test
from models import helper
import commons

import pandas as pd

from utils.checkpoint import resume_model_with_classifiers, resume_train_with_groups
from utils.inference import inference_with_groups, inference_with_groups_csv
from utils.experiment import build_classifiers, build_test_dataset, get_dataset_split_names, make_backbone_info
from utils.utils import move_to_device, get_utms_from_paths, get_utm_from_path
from models.classifiers import AAMC, QAMC, LMCC, LinearLayer
import parser

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

assert args.train_set_path is not None, 'you must specify the train set path'
assert os.path.exists(args.train_set_path), 'train set path must exist'
assert (args.test_set_path is not None) or (args.val_set_path is not None), 'you must specify the test set path'
if args.test_set_path is not None:
    assert os.path.exists(args.test_set_path), 'test set path must exist'
if args.val_set_path is not None:
    assert os.path.exists(args.val_set_path), 'val set path must exist (real photo)'



train_dataset_folders, test_datasets = get_dataset_split_names(args)
backbone_info = make_backbone_info(args)
agg_config = {"mixvpr_out_channels": args.mixvpr_out_channels}


train_transform = T.Compose([
    T.Resize(args.train_resize, antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform =T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Initialization
commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

# Datasets and dataloaders
groups = []
for n in range(args.N):
    group = HCDataset_shN(group_num=n, dataset_name=args.dataset_name,train_path=args.train_set_path, train_dataset_folders=train_dataset_folders, M=args.M, N=args.N,min_images_per_class=args.min_images_per_class,transform=train_transform)
    groups.append(group)

test_dataset = build_test_dataset(args, test_datasets, test_transform)

test_img_num = len(test_dataset)
logging.info(f'Found {test_img_num} images in the test set.' )
test_num_workers = 2 if (args.device == "cuda" and platform.system() == "Linux") else 0
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=test_num_workers, pin_memory=(args.device == "cuda"))

# Model
model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config)

model = model.to(args.device)

# Build one classifier per altitude group.
classifiers = build_classifiers(args, model.feature_dim, groups)
classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")
logging.info(f"Feature dim: {model.feature_dim}")
logging.info(f"resume_model: {args.resume_model}")


# Count trainable parameters.
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(f'Trainable parameters: {params/1e6:.4}M')

# Optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True)

# Resume
if args.resume_model is not None:
    model, classifiers = resume_model_with_classifiers(model, classifiers)

if args.resume_train is not None:
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
        resume_train_with_groups(args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = 0
    start_epoch_num = 0
    best_loss = float('inf')

images_info = inference_with_groups_csv(args, model, classifiers, test_dl, groups, test_img_num)

for image_info in images_info:
    utm = get_utm_from_path(image_info['image_path'])
    image_info['utm_e'] = utm[0]
    image_info['utm_n'] = utm[1]

df = pd.DataFrame(images_info)
df.to_csv(os.path.join(args.save_dir, 'results.csv'), index=False)


correct_class_recall, threshold_recall = inference_with_groups(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

logging.info(f"Test LR: {correct_class_recall}, {threshold_recall}")

cross_entropy_loss = torch.nn.CrossEntropyLoss()


torch.cuda.empty_cache()
