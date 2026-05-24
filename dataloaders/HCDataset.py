from __future__ import annotations

import bisect
import hashlib
import logging
import os
import random
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

import parser
from utils.pipeline_ops import ALTITUDE_INTERVAL, H_MIN
from utils.utils import tensor_fft_3D


args = parser.parse_arguments()

DEFAULT_IMAGE_SIZE = (360, 480)

basic_transform = T.Compose(
    [
        T.Resize(DEFAULT_IMAGE_SIZE, antialias=True),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get__class_id__group_id(height, M, N):
    class_id = int(height // M * M)
    group_id = class_id % (M * N) // M
    return class_id, group_id


def get_heights_from_paths(images_paths: list[str]):
    heights = []
    for image_path in images_paths:
        info = image_path.split("/")[-1].split("@")
        if len(info[-1]) > 4:
            heights.append(float(info[4]))
        else:
            heights.append(float(info[2]))
    return heights


def get_heights_from_qingdao_paths(images_paths: list[str]):
    info_list = [image_path.split("/")[-1].split("@") for image_path in images_paths]
    return [float(info[4]) for info in info_list]


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def initialize(dataset_folder, train_dataset_folders, dataset_name, M, N, min_images_per_class):
    cache_key = hashlib.md5(
        "|".join([os.path.abspath(dataset_folder), ",".join(train_dataset_folders), str(dataset_name)]).encode("utf-8")
    ).hexdigest()[:10]
    paths_file = f"cache/paths_{dataset_name}_{cache_key}_M{M}_N{N}_mipc{min_images_per_class}.torch"

    if not os.path.exists(paths_file):
        logging.info("Searching training images in %s", dataset_folder)
        images_paths = []
        for train_dataset_folder in train_dataset_folders:
            images_paths.extend(sorted(glob(f"{dataset_folder}/{train_dataset_folder}/**/*.png", recursive=True)))
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]

        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
    else:
        images_paths = torch.load(paths_file)

    logging.info("Found %d images", len(images_paths))

    heights = get_heights_from_paths(images_paths)
    class_id__group_id = [get__class_id__group_id(h, M, N) for h in heights]

    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)

    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id in images_per_class:
            classes_per_group[group_id].add(class_id)

    for group_id, group_classes in classes_per_group.items():
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id]

    classes_per_group = [sorted(list(c)) for c in classes_per_group.values()]
    images_per_class_per_group = [
        {k: v for k, v in sorted(subdict.items())}
        for subdict in images_per_class_per_group.values()
    ]

    ends_per_group = []
    for images_per_class_in_current_group in images_per_class_per_group:
        image_counts = [len(paths) for paths in images_per_class_in_current_group.values()]
        ends_per_group.append([sum(image_counts[: i + 1]) for i in range(len(image_counts))])

    return classes_per_group, images_per_class_per_group, ends_per_group


class HCDataset_shN(Dataset):
    def __init__(
        self,
        group_num,
        dataset_name,
        train_path,
        train_dataset_folders,
        M=args.M,
        N=args.N,
        min_images_per_class=15,
        transform=basic_transform,
    ):
        super().__init__()

        cache_key = hashlib.md5(
            "|".join([os.path.abspath(train_path), ",".join(train_dataset_folders), str(dataset_name)]).encode("utf-8")
        ).hexdigest()[:10]
        cache_filename = f"cache/{dataset_name}_{cache_key}_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group, ends_per_group = initialize(
                train_path,
                train_dataset_folders,
                dataset_name,
                M,
                N,
                min_images_per_class,
            )
            torch.save((classes_per_group, images_per_class_per_group, ends_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group, ends_per_group = torch.load(cache_filename)

        self.train_path = train_path
        self.M = M
        self.N = N
        self.transform = transform
        self.classes_ids = classes_per_group[group_num]
        self.images_per_class = images_per_class_per_group[group_num]
        self.class_centers = [class_id + M // 2 for class_id in self.classes_ids]
        self.classes_num_total = len(self.classes_ids)
        self.ends = ends_per_group[group_num]
        self.fft = args.fft
        self.fft_log_base = args.fft_log_base
        self.group_len = self.get_images_num()

    def __getitem__(self, index):
        class_num_current = bisect.bisect_right(self.ends, index)
        assert class_num_current < self.classes_num_total, "class_num_current >= classes_num_total"

        class_id_current = self.classes_ids[class_num_current]
        class_center_current = self.class_centers[class_num_current]
        image_path = self.train_path + random.choice(self.images_per_class[class_id_current])

        try:
            image = Image.open(image_path).convert("RGB")
        except UnidentifiedImageError:
            logging.info("Failed to read image %s; using a zero tensor instead", image_path)
            image = torch.zeros([3, args.train_resize[0], args.train_resize[1]])

        if self.transform:
            image = self.transform(image)
        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)

        return image, class_num_current, class_center_current

    def get_images_num(self):
        return sum(len(self.images_per_class[class_id]) for class_id in self.classes_ids)

    def get_classes_num(self):
        return len(self.classes_ids)

    def __len__(self):
        return self.group_len


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_folder, test_datasets, M=10, N=5, image_size=256):
        super().__init__()
        images_paths = []
        for dataset in test_datasets:
            images_paths.extend(sorted(glob(f"{test_folder}/{dataset}**/*.png", recursive=True)))

        self.heights = get_heights_from_paths(images_paths)
        class_id_group_id = [get__class_id__group_id(h, M, N) for h in self.heights]
        self.images_paths = images_paths
        self.class_centers = [class_id + M // 2 for class_id, _ in class_id_group_id]
        self.class_id = [class_id for class_id, _ in class_id_group_id]
        self.group_id = [group_id for _, group_id in class_id_group_id]
        self.normalize = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = self.normalize(Image.open(image_path).convert("RGB"))
        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        return image, self.class_id[index], self.heights[index], image_path

    def __len__(self):
        return len(self.images_paths)


class QingdaoFlightDataset(torch.utils.data.Dataset):
    def __init__(self, test_folder, dataset_name=None, M=10, N=5, image_size=256):
        super().__init__()
        dataset_root = Path(test_folder)
        if dataset_name is not None and (dataset_root / dataset_name).exists():
            dataset_root = dataset_root / dataset_name

        images_paths = sorted(glob(f"{dataset_root}/**/*.png", recursive=True))
        self.heights = get_heights_from_qingdao_paths(images_paths)
        class_id_group_id = [get__class_id__group_id(h, M, N) for h in self.heights]
        self.images_paths = images_paths
        self.class_centers = [class_id + M // 2 for class_id, _ in class_id_group_id]
        self.class_id = [class_id for class_id, _ in class_id_group_id]
        self.group_id = [group_id for _, group_id in class_id_group_id]
        self.normalize = T.Compose(
            [
                T.Resize(image_size, antialias=True),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image = self.normalize(Image.open(image_path).convert("RGB"))
        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        return image, self.class_id[index], self.heights[index], image_path

    def __len__(self):
        return len(self.images_paths)
