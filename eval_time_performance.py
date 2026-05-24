import warnings
import os
import sys
import platform
import psutil
import time
import logging
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as T

warnings.filterwarnings("ignore", category=UserWarning)

import parser
import commons
from dataloaders.HCDataset import HCDataset_shN
from utils.checkpoint import resume_model_with_classifiers
from utils.experiment import build_classifiers, build_test_dataset, get_dataset_split_names, make_backbone_info
from utils.inference import inference_latency_memory, inference_with_groups_csv
from utils.utils import get_utm_from_path
from models.classifiers import AAMC, QAMC, LMCC, LinearLayer
from models import helper



os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

assert args.train_set_path is not None, 'you must specify the train set path'
assert os.path.exists(args.train_set_path), 'train set path must exist'
assert args.test_set_path is not None, 'you must specify the test set path'
assert os.path.exists(args.test_set_path), 'test set path must exist'



train_dataset_folders, test_datasets = get_dataset_split_names(args)
backbone_info = make_backbone_info(args)
agg_config = {"mixvpr_out_channels": args.mixvpr_out_channels}


train_transform = T.Compose([
    T.Resize(args.train_resize, antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = T.Compose([
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

# Resume
if args.resume_model is not None:
    model, classifiers = resume_model_with_classifiers(model, classifiers)
else:
    raise ValueError("No model to resume.")

avg_latency, max_latency, cpu_peak, cpu_delta, gpu_peak = inference_latency_memory(
    args=args,
    model=model,
    classifiers=classifiers,
    test_dl=test_dl,
    groups=groups,
    num_test_images=test_img_num
)


logging.info("Performance Metrics:")
logging.info(f"Avg latency: {avg_latency:.2f}ms")
logging.info(f"Max latency: {max_latency:.2f}ms")
logging.info(f"Peak CPU ABS: {cpu_peak:.2f}MB")
logging.info(f"CPU DELTA: {cpu_delta:.2f}MB")
logging.info(f"Peak GPU: {gpu_peak:.2f}MB")
