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

from dataloaders.HCDataset import InfiniteDataLoader, HCDataset_shN
from models import helper
import commons

from utils.checkpoint import resume_model_with_classifiers, resume_train_with_groups_all, save_checkpoint_with_groups_best_val
from utils.inference import inference_with_groups, inference_with_groups_with_val
from utils.experiment import build_classifiers, build_test_dataset, classifier_forward, get_dataset_split_names, make_backbone_info
from utils.utils import move_to_device
from models.classifiers import AAMC, QAMC, LMCC, LinearLayer
import parser

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

logging.info(f"train_dataset_folders: {train_dataset_folders}")
logging.info(f"test_datasets_folders: {test_datasets}")

# Datasets and dataloaders
groups = []
for n in range(args.N):
    group = HCDataset_shN(group_num=n, dataset_name=args.dataset_name, train_path=args.train_set_path, train_dataset_folders=train_dataset_folders, M=args.M, N=args.N,min_images_per_class=args.min_images_per_class, transform=train_transform)
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
elif args.resume_train is not None:
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num, scheduler, best_val_lr = \
        resume_train_with_groups_all(args.save_dir, model, optimizer, classifiers, classifiers_optimizers, scheduler)
    
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    best_val = best_val_lr
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = 0
    start_epoch_num = 0
    best_loss = float('inf')
    best_val = 0.0


# Training loss and mixed precision state.
cross_entropy_loss = torch.nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler(enabled=args.device == "cuda")
for epoch_num in range(start_epoch_num, args.epochs_num):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break

    classes_num = 0
    for g in groups:
        classes_num += g.get_classes_num()
    classes_num_list = [g.get_classes_num() for g in groups]

    train_loss = torchmetrics.MeanMetric().to(args.device)

    current_group_num = epoch_num % len(classifiers)
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num_list[current_group_num]).to(args.device)
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    move_to_device(classifiers_optimizers[current_group_num], args.device)

    # Train the group selected for this epoch.
    train_dataloader = InfiniteDataLoader(groups[current_group_num],
                                          num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=True,
                                          pin_memory=(args.device == "cuda"), drop_last=True)
    dataloader_iterator = iter(train_dataloader)
    model = model.train()

    tqdm_bar = tqdm(range(args.iterations_per_epoch), ncols=100, desc="")
    for iteration in tqdm_bar:
        images, labels, _ = next(dataloader_iterator)
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        with torch.autocast("cuda", enabled=args.device == "cuda"):
            descriptors = model(images)
            output, logits = classifier_forward(classifiers[current_group_num], descriptors, labels, images=images)
            loss = cross_entropy_loss(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(classifiers_optimizers[current_group_num])
        scaler.update()

        train_acc.update(logits, labels)
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        del loss, images, output

    # Validation
    correct_class_recall, threshold_recall, val_lr = inference_with_groups_with_val(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

    train_acc = train_acc.compute() * 100  
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False

    if val_lr > best_val:
        is_best_val = True
        best_val = val_lr
    else:
        is_best_val = False
        

    logging.info(f"E{epoch_num: 3d}, train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{args.scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}, " +
                 f"classifier_lr: {round(classifiers_optimizers[current_group_num].param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch_num: 3d}, {correct_class_recall}")
    logging.info(f"E{epoch_num: 3d}, {threshold_recall}")

    scheduler.step(train_loss)

    save_checkpoint_with_groups_best_val({"epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "args": args,
        "best_train_loss": best_loss,
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_lr": best_val,
    }, is_best, is_best_val, args.save_dir)


    torch.cuda.empty_cache()

correct_class_recall, threshold_recall = inference_with_groups(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

logging.info(f"Test LR: {correct_class_recall}, {threshold_recall}")
print("Training complete.")
