# https://github.com/amaralibey/gsv-cities

import pandas as pd
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import torch
import torch.utils
from torch.utils.data import Dataset
import torch.utils.data
import torchvision.transforms as T
from glob import glob
import numpy as np
from utils.utils import image_fft, tensor_to_image, tensor_fft_3D

import os
import parser
import random
import logging
from collections import defaultdict
import bisect
from hc_db_cut import h_max, h_min, class_interval, classes_num

import math
import hashlib

size = (360, 480)

# 给图像加变形，看是不是能训练出结果

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



# Default paths are only used when CLI paths are omitted.
args = parser.parse_arguments()

if args.train_set_path is not None:
    BASE_PATH = args.train_set_path
else:
    # BASE_PATH = '/root/workspace/maps/HE-100-700/'
    BASE_PATH = '/root/workspace/maps/HC-100-700/'

if args.test_set_path is not None:
    TEST_PATH = args.test_set_path
else:
    # TEST_PATH = '/root/workspace/maps/HE-100-700-test/'
    TEST_PATH = '/root/workspace/maps/HC-100-700-test/'

if args.val_set_path is not None:
    real_BASE_PATH = args.val_set_path
else:
    real_BASE_PATH = '/root/workspace/maps/HE_Test/'
    


# BASE_PATH = '/root/workspace/maps/HE-100-700/'
# real_BASE_PATH = '/root/workspace/maps/HE_Test/'
# TEST_PATH = '/root/workspace/maps/HE-100-700-test/'

# if not Path(BASE_PATH).exists():
#     raise FileNotFoundError(
#         'BASE_PATH is hardcoded, please adjust to point to gsv_cities')

default_transform = T.Compose([
    T.Resize(size, antialias=True),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
    T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

basic_transform = T.Compose([
    T.Resize(size, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get__class_id__group_id(height, M, N):
    """Return class_id and group_id for a given point.
        The class_id is a tuple of UTM_east, UTM_north (e.g. (396520, 4983800)).
        The group_id represents the group to which the class belongs
        (e.g. (0, 1), and it is between (0, 0) and (N, N).
    """
    class_id = int(height // M * M)  # Rounded to nearest lower multiple of M

    # group_id goes from (0, 0) to (N, N)
    group_id = (class_id % (M * N) // M)
    return class_id, group_id

# def get__class_id__group_id(height, M=1, N=1, H0=100, delta0=20, r=1.1):
#     """
#     Return class_id and group_id for a given height using exponential/geometric widening.
    
#     Parameters:
#         height : float, the altitude to classify
#         H0     : float, starting altitude (lower bound)
#         delta0 : float, initial interval size
#         r      : float, geometric ratio (>1)
#         N      : int, number of groups (default=1)
    
#     Returns:
#         class_id : int, the bin index
#         group_id : int, the group index
#     """
#     if height < H0:
#         class_id = -1  # 低于起始高度，标记为无效
#     else:
#         val = 1 + (height - H0) * (r - 1) / delta0
#         class_id = math.floor(math.log(val, r)) if val > 0 else 0
    
#     group_id = class_id % N if class_id != -1 else -1
#     return class_id, group_id

def get_heights_from_paths(images_paths: list[str]):

    # images_metadatas = [p.split("@") for p in images_paths]
    # heights = [m[2] for m in images_metadatas]
    # # heights = np.array(heights).astype(np.float64)
    # del images_metadatas

    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
    # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
    # heights = [float(info[3]) for info in info_list]    # NOTE 给VPR切的图像是filename = f'@{year}@{rotation_angle}@{flight_height}@{CT_utm_e}@{CT_utm_n}@.png'，应该是第三个

    # heights = [float(info[2]) for info in info_list]
    heights = []
    for info in info_list:
        if len(info[-1]) > 4:
            heights.append((float(info[4])))
        else:
            heights.append(float(info[2]))
    return heights

def get_heights_from_qingdao_paths(images_paths: list[str]):

    # images_metadatas = [p.split("@") for p in images_paths]
    # heights = [m[2] for m in images_metadatas]
    # # heights = np.array(heights).astype(np.float64)
    # del images_metadatas

    info_list = [image_path.split('/')[-1].split('@') for image_path in images_paths]
    # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
    # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
    heights = [float(info[4]) for info in info_list]
    # heights = [float(info[3]) for info in info_list]    # NOTE 给VPR切的图像是filename = f'@{year}@{rotation_angle}@{flight_height}@{CT_utm_e}@{CT_utm_n}@.png'，应该是第三个
    return heights

class HCDataset(Dataset):
    def __init__(self,
                 foldernames=['2022'],
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                #  random_transform=True,
                 ):
        super(HCDataset, self).__init__()
        self.base_path = base_path
        self.foldernames = foldernames

        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base
        
        # generate the dataframe contraining images metadata
        self.dataframes = self.__getdataframes()
        self.year = list(self.dataframes['year'])
        self.flight_height = list(self.dataframes['flight_height'])
        self.heights_tensor = torch.tensor(self.flight_height).unsqueeze(1)
        self.alpha = list(self.dataframes['rotation_angle'])
        self.loc_x = list(self.dataframes['loc_x'])
        self.loc_y = list(self.dataframes['loc_y'])
        self.heights_labels = list(self.dataframes['flight_class'])
        # self.random_transform = random_transform
        
        # get all unique place ids
        self.total_nb_images = len(self.dataframes)
        
    def __getdataframes(self) -> pd.DataFrame:
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        csv_path = os.path.join(self.base_path, 'Dataframes', f'{self.foldernames[0]}.csv')
        df = pd.read_csv(csv_path, encoding='utf-8', converters = {'year':str})
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.foldernames)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.foldernames[i]}.csv', encoding='utf-8', converters = {'year':str})

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        if self.random_sample_from_each_place:
            df = df.sample(frac=1)

        # keep only places depicted by at least min_img_per_place images
        # res = df[df.groupby('place_id')['place_id'].transform('size') >= self.min_img_per_place]
        # return res.set_index('place_id')

        return df.reset_index(drop=True)
    
    def __getitem__(self, index):
        
        height = self.heights_tensor[index]
        height_label = int(self.heights_labels[index])
        image_name = f'@{self.year[index]}@{self.flight_height[index]:.2f}@{self.alpha[index]:.2f}@{self.loc_x[index]}@{self.loc_y[index]}@.png'
        # NOTE f'{year}@{flight_height:.2f}@{alpha:.2f}@{loc_w}@{loc_h}.png'
        image_path = os.path.join(self.base_path, 'Images', self.year[index], image_name)
        image = Image.open(image_path).convert('RGB')
        
        # if self.random_transform:
        #     # image_new, height_new = random_transform(image, height)
        #     # h, w = image.shape[1], image.shape[2]
        #     h = image.height
        #     w = image.width
        #     scale = random.uniform(1.4, 2)
        #     crop_w = w // scale
        #     crop_h = h // scale
        #     real_scale = ((h / crop_h) + (w / crop_w))/2
        #     flip_chance = 0.5
        #     rand_transform = T.Compose([
        #         T.CenterCrop((crop_h, crop_w)),
        #         T.Resize((h, w), resize()),
        #     ])
        #     height_new = height/real_scale
        #     image_new = rand_transform(image)
        #     # image_out = [image, image_new]
        #     # height_out = [height, height_new]
            
        #     if self.transform:
        #         image, image_new = self.transform(image), self.transform(image_new)
            
        #     image_out = torch.stack((image, image_new), dim=0)
        #     height_out = torch.stack((height, height_new), dim=0)
        
        #     return image_out, height_label
        
        if self.transform:
            image = self.transform(image)
        
        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)

        class_center = height_label * class_interval + h_min + class_interval * 0.5

        # return image, height_label, class_center
        return image, height_label, height
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.flight_height)



class realHCDataset(Dataset):
    def __init__(self, base_path=real_BASE_PATH, transform=basic_transform):
        super().__init__()
        self.base_path = base_path

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

        self.transform = transform
        images_paths = sorted(glob(f"{base_path}/**/*.png", recursive=True))
        
        self.images_paths = images_paths
        self.heights = self.get_heights(images_paths)

    def __getitem__(self, index):
        
        height = self.heights[index]
        height_label = (height - h_min) // class_interval

        # height = scale_down(height) # NOTE: 后期加入归一化，方便收敛
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        return image, height_label, height
    
    def __len__(self):
        return len(self.images_paths)
    
    @staticmethod
    def get_heights(image_paths):
        info_list = [image_path.split('/')[-1].split('@') for image_path in image_paths]
        # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
        # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
        heights = torch.tensor([float(info[-4]) for info in info_list])
        return heights


class testHCDataset(Dataset):
    def __init__(self,
                 foldernames=['2022'],
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=TEST_PATH,
                 ):
        super().__init__()
        self.base_path = base_path
        self.foldernames = foldernames

        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base
        
        # generate the dataframe contraining images metadata
        self.dataframes = self.__getdataframes()
        self.year = list(self.dataframes['year'])
        self.flight_height = list(self.dataframes['flight_height'])
        self.heights_tensor = torch.tensor(self.flight_height).unsqueeze(1)
        self.alpha = list(self.dataframes['rotation_angle'])
        self.loc_x = list(self.dataframes['loc_x'])
        self.loc_y = list(self.dataframes['loc_y'])
        self.heights_labels = list(self.dataframes['flight_class'])
        
    def __getdataframes(self) -> pd.DataFrame:
        ''' 
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        csv_path = os.path.join(self.base_path, 'Dataframes', f'{self.foldernames[0]}.csv')
        df = pd.read_csv(csv_path, encoding='utf-8', converters = {'year':str})
        df = df.sample(frac=1)  # shuffle the city dataframe
        

        # append other cities one by one
        for i in range(1, len(self.foldernames)):
            csv_path = os.path.join(self.base_path, 'Dataframes', f'{self.foldernames[i]}.csv')
            tmp_df = pd.read_csv(
                csv_path, encoding='utf-8', converters = {'year':str})

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            # TODO: rename the dataset and hardcode these prefixes
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            
            df = pd.concat([df, tmp_df], ignore_index=True)

        if self.random_sample_from_each_place:
            df = df.sample(frac=1)

        # keep only places depicted by at least min_img_per_place images
        # res = df[df.groupby('place_id')['place_id'].transform('size') >= self.min_img_per_place]
        # return res.set_index('place_id')

        return df.reset_index(drop=True)
    
    def __getitem__(self, index):
        
        height = self.heights_tensor[index]
        height_label = self.heights_labels[index]
        image_name = f'@{self.year[index]}@{self.flight_height[index]:.2f}@{self.alpha[index]:.2f}@{self.loc_x[index]}@{self.loc_y[index]}@.png'        
        image_path = os.path.join(self.base_path, 'Images', self.year[index], image_name)
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)

        return image, height_label, height
        # return image, height_label
    
    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.flight_height)

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

class HCDataset_sh(Dataset):
    def __init__(self, base_path=real_BASE_PATH, transform=basic_transform):
        super().__init__()
        self.base_path = base_path

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

        self.transform = transform
        images_paths = sorted(glob(f"{base_path}/**/*.png", recursive=True))
        
        self.images_paths = images_paths
        self.heights = self.get_heights(images_paths)

    def __getitem__(self, index):
        
        height = self.heights[index]
        height_label = (height - h_min) // class_interval

        # height = scale_down(height) # NOTE: 后期加入归一化，方便收敛
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        return image, height_label, height
    
    def __len__(self):
        return len(self.images_paths)
    
    @staticmethod
    def get_heights(image_paths):
        info_list = [image_path.split('/')[-1].split('@') for image_path in image_paths]
        # heights = np.array([info[-4] for info in info_list]).astype(np.float16)
        # heights = torch.tensor(heights, dtype=torch.float16).unsqueeze(1)
        heights = torch.tensor([float(info[-4]) for info in info_list])
        return heights


def initialize(dataset_folder, train_dataset_folders, dataset_name, M, N, min_images_per_class):
    cache_key = hashlib.md5(
        "|".join([os.path.abspath(dataset_folder), ",".join(train_dataset_folders), str(dataset_name)]).encode("utf-8")
    ).hexdigest()[:10]
    paths_file = f"cache/paths_{dataset_name}_{cache_key}_M{M}_N{N}_mipc{min_images_per_class}.torch"
    # Search paths of dataset only the first time, and save them in a cached file
    if not os.path.exists(paths_file):
        logging.info(f"Searching training images in {dataset_folder}")
        images_paths = []
        for train_dataset_folder in train_dataset_folders:
            images_paths_current_folder = sorted(glob(f"{dataset_folder}/{train_dataset_folder}/**/*.png", recursive=True))
            images_paths.extend(images_paths_current_folder)
        # Remove folder_path from images_path, so that the same cache file can be used on any machine
        images_paths = [p.replace(dataset_folder, "") for p in images_paths]
        
        os.makedirs("cache", exist_ok=True)
        torch.save(images_paths, paths_file)
    else:
        images_paths = torch.load(paths_file)

    logging.info(f"Found {len(images_paths)} images")

    heights = get_heights_from_paths(images_paths)

    logging.info("For each image, get its flight height from its path")
    logging.info("For each image, get class and group to which it belongs")
    class_id__group_id = [get__class_id__group_id(h, M, N) for h in heights]

    logging.info("Group together images belonging to the same class")
    images_per_class = defaultdict(list)
    images_per_class_per_group = defaultdict(dict)
    for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
        images_per_class[class_id].append(image_path)

    # Images_per_class is a dict where the key is class_id, and the value
    # is a list with the paths of images within that class.
    images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

    logging.info("Group together classes belonging to the same group")
    # Classes_per_group is a dict where the key is group_id, and the value
    # is a list with the class_ids belonging to that group.
    classes_per_group = defaultdict(set)
    for class_id, group_id in class_id__group_id:
        if class_id not in images_per_class:
            continue  # Skip classes with too few images
        classes_per_group[group_id].add(class_id)   # 每个group_id对应的class_id可以保证没有重复元素，格式：defaultdict(<class 'set'>, {0: {0,2,4}})

    for group_id, group_classes in classes_per_group.items():   # NOTE 这里得到的group_id是一个数，group_classes是一个set，格式：dict_items([(0, {0, 20}), (1, {1, 2})])，group_id和group_classes分别对应，0和{0, 20}，1和{1, 2}
        for class_id in group_classes:
            images_per_class_per_group[group_id][class_id] = images_per_class[class_id] # NOTE images_per_class_per_group 格式：defaultdict(<class 'dict'>, {0: {0: 'power.png'}, 1: {1: 'pwr.png'}})
    # Convert classes_per_group to a list of lists.
    # Each sublist represents the classes within a group.
    classes_per_group = [list(c) for c in classes_per_group.values()]   # classes_per_group.values()的格式：dict_values([{0, 20}, {1, 2}])；list(c)的格式：[0, 20], classes_per_group格式：[[0, 20], [1, 2]]
    classes_per_group= [sorted(sublist) for sublist in classes_per_group]   # group的sublist中，class的数值从小到大排序
    images_per_class_per_group = [c for c in images_per_class_per_group.values()]   # NOTE images_per_class_per_group 格式：[{0: ['power.png']}, {1: ['pwr.png']}, {2: ['1.png', '2.png'], 1: ['4.png', '5.png']}]
    images_per_class_per_group = [{k: v for k, v in sorted(subdict.items())} for subdict in images_per_class_per_group] # [{100: ['/ct02/0100/@ct02@100.00@00@0.00@0@0@.png', ...], 200: [...], 300: [...], 400: [...], 500: [...], 600: [...]}, {450: [...], 550: [...], 650: [...], 150: [...], 250: [...], 350: [...]}]

    images_num_per_class_per_group = []
    ends_per_group = []
    for group in range(len(images_per_class_per_group)):
        images_per_class_in_current_group = images_per_class_per_group[group]
        images_num_per_class_in_current_group = defaultdict(int)
        for class_id, images_paths_list in images_per_class_in_current_group.items():
            images_num_per_class_in_current_group[class_id] = len(images_paths_list)
        images_num_per_class_per_group.append(images_num_per_class_in_current_group)
        images_num_per_class_in_current_group_list = list(images_num_per_class_in_current_group.values())
        ends = [sum(images_num_per_class_in_current_group_list[:i+1]) for i in range(len(images_num_per_class_in_current_group))]

        ends_per_group.append(ends)

    return classes_per_group, images_per_class_per_group, ends_per_group

class HCDataset_shN(Dataset):
    def __init__(self, group_num, dataset_name, train_path, train_dataset_folders, M = args.M, N = args.N, min_images_per_class = 15,transform=basic_transform):
        super().__init__()


        cache_key = hashlib.md5(
            "|".join([os.path.abspath(train_path), ",".join(train_dataset_folders), str(dataset_name)]).encode("utf-8")
        ).hexdigest()[:10]
        cache_filename = f"cache/{dataset_name}_{cache_key}_M{M}_N{N}_mipc{min_images_per_class}.torch"
        if not os.path.exists(cache_filename):
            classes_per_group, images_per_class_per_group, ends_per_group = initialize(train_path, train_dataset_folders, dataset_name, M, N, min_images_per_class)
            torch.save((classes_per_group, images_per_class_per_group, ends_per_group), cache_filename)
        else:
            classes_per_group, images_per_class_per_group, ends_per_group = torch.load(cache_filename)
        classes_ids = classes_per_group[group_num]
        images_per_class = images_per_class_per_group[group_num]    # 当前group中所有class的图片相对train_path的路径

        ends = ends_per_group[group_num]

        # self.transform = transform
        # images_paths = sorted(glob(f"{base_path}/**/*.png", recursive=True))
        
        # self.images_paths = images_paths
        # self.heights = self.get_heights(images_paths)

        self.train_path = train_path

        self.M = M
        self.N = N
        self.transform = transform
        self.classes_ids = classes_ids  # classes_ids是一个list，格式：[0, 20]，代表当前group中的所有class的id
        self.images_per_class = images_per_class
        self.class_centers = [cl_id + M // 2 for cl_id in self.classes_ids]
        # # -------------------------- 修改 class_centers 定义 --------------------------
        # # 在 __init__ 的 class_centers 定义处
        # # 几何区间参数（与 get__class_id__group_id 一致）
        # H0 = 100
        # delta0 = 20
        # r = 1.1

        # self.class_centers = []
        # for cl_id in self.classes_ids:
        #     if cl_id == -1:
        #         self.class_centers.append(-1.0)
        #     else:
        #         lower = H0 + delta0 * (r**cl_id - 1) / (r - 1)
        #         upper = H0 + delta0 * (r**(cl_id + 1) - 1) / (r - 1)
        #         center = (lower + upper) / 2
        #         self.class_centers.append(round(center, 2))
        # logging.info(f'class_centers: {self.class_centers}')
        # logging.info(f'classes_ids: {self.classes_ids}')
        # # ---------------------------------------------------------------------------
        self.classes_num_total = len(classes_ids)

        self.ends = ends

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

        self.group_len = self.get_images_num()

    def __getitem__(self, index):

        class_num_current = bisect.bisect_right(self.ends, index)
        assert class_num_current < self.classes_num_total, 'class_num_current >= classes_num_total'
        # class_num_current = random.randint(0, self.classes_num_total - 1)
        class_id_current = self.classes_ids[class_num_current]
        class_center_current = self.class_centers[class_num_current]

        image_path = self.train_path + random.choice(self.images_per_class[class_id_current])
        
        try:
            image = Image.open(image_path).convert('RGB')
        except UnidentifiedImageError:
            logging.info(f"ERR: There was an error while reading image {image_path}, it is probably corrupted")
            image = torch.zeros([3, args.train_resize[0], args.train_resize[1]])
        
        if self.transform:
            image = self.transform(image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        return image, class_num_current, class_center_current
    
    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])
    
    def get_classes_num(self):
        """Return the number of classes within this group."""
        return len(self.classes_ids)
    
    def __len__(self):
        """Return a large number. This is because if you return the number of
        classes and it is too small (like in pitts30k), the dataloader within
        InfiniteDataLoader is often recreated (and slows down training).
        """
        return self.group_len
        return 1000000
    
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, test_folder, test_datasets, M=10, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        images_paths = []
        for dataset in test_datasets:
            images_paths_current_folder = sorted(glob(f"{test_folder}/{dataset}**/*.png", recursive=True))
            images_paths.extend(images_paths_current_folder)

        logging.debug(f"Found {len(images_paths)} images")

        self.heights = get_heights_from_paths(images_paths)

        class_id_group_id = [get__class_id__group_id(h, M, N) for h in self.heights]    # 得到(class_id, group_id)
        self.images_paths = images_paths
        self.class_centers = [id[0] + M // 2 for id in class_id_group_id]
        self.class_id = [id[0] for id in class_id_group_id]
        self.group_id = [id[1] for id in class_id_group_id]
        # # -------------------------- 修改 class_centers 定义 --------------------------
        # # 在 __init__ 的 class_centers 定义处
        # # 几何区间参数（与 get__class_id__group_id 一致）
        # H0 = 100
        # delta0 = 20
        # r = 1.1

        # self.class_centers = []
        # for cl_id in self.class_id:
        #     if cl_id == -1:
        #         self.class_centers.append(-1.0)
        #     else:
        #         lower = H0 + delta0 * (r**cl_id - 1) / (r - 1)
        #         upper = H0 + delta0 * (r**(cl_id + 1) - 1) / (r - 1)
        #         center = (lower + upper) / 2
        #         self.class_centers.append(round(center, 2))
        # # ---------------------------------------------------------------------------

        self.normalize = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        # class_id = self.class_id[index]

        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        # if isinstance(image, tuple):
        #     image = torch.stack(image, dim=0)
        return image, self.class_id[index], self.heights[index], image_path

    def __len__(self):
        return len(self.images_paths)
class realHCDataset_N(Dataset):
    def __init__(self, base_path, M, N, transform=basic_transform):
        super().__init__()
        self.base_path = base_path

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

        self.transform = transform

        subdirs = []
        true_heights = []
        labels = [0, 1]
        known_subdirs = ['GeoVINS_VPR', 'GeoVINS_VPR2','GeoVINS_VPR_h400', 'GeoVINS_VPR_h630']
        known_heights = [150, 175, 400, 630]
        subdirs = known_subdirs
            
        images_paths = []
        heights = []
        for i in range(len(subdirs)):
            subdir = subdirs[i]
            images_paths_h = sorted(glob(f"{base_path}/{subdir}/*.png", recursive=True))
            images_paths.extend(images_paths_h)
            images_num = len(images_paths_h)
            heights.extend([known_heights[i]] * images_num)

        self.images_paths = images_paths
        # self.heights = self.get_heights(images_paths)
        self.heights = heights

        class_id_group_id = [get__class_id__group_id(h, M, N) for h in heights]    # 得到(class_id, group_id)
        self.images_paths = images_paths
        self.class_centers = [id[0] + M // 2 for id in class_id_group_id]
        self.class_id = [id[0] for id in class_id_group_id]
        self.group_id = [id[1] for id in class_id_group_id]

    def __getitem__(self, index):
        
        height = self.heights[index]
        # height_label = (height - h_min) // class_interval



        # height = scale_down(height) # NOTE: 后期加入归一化，方便收敛
        image_path = self.images_paths[index]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        return image, self.class_id[index], self.heights[index], image_path
    
    def __len__(self):
        return len(self.images_paths)


class TestDatasetNew(torch.utils.data.Dataset):
    def __init__(self, test_folder, M=10, N=5, image_size=256):
        super().__init__()
        logging.debug(f"Searching test images in {test_folder}")

        images_paths = sorted(glob(f"{test_folder}/**/*.png", recursive=True))

        logging.debug(f"Found {len(images_paths)} images")

        self.heights = get_heights_from_qingdao_paths(images_paths)

        class_id_group_id = [get__class_id__group_id(h, M, N) for h in self.heights]    # 得到(class_id, group_id)
        self.images_paths = images_paths
        self.class_centers = [id[0] + M // 2 for id in class_id_group_id]
        self.class_id = [id[0] for id in class_id_group_id]
        self.group_id = [id[1] for id in class_id_group_id]
        # # -------------------------- 修改 class_centers 定义 --------------------------
        # # 在 __init__ 的 class_centers 定义处
        # # 几何区间参数（与 get__class_id__group_id 一致）
        # H0 = 100
        # delta0 = 20
        # r = 1.1

        # self.class_centers = []
        # for cl_id in self.class_id:
        #     if cl_id == -1:
        #         self.class_centers.append(-1.0)
        #     else:
        #         lower = H0 + delta0 * (r**cl_id - 1) / (r - 1)
        #         upper = H0 + delta0 * (r**(cl_id + 1) - 1) / (r - 1)
        #         center = (lower + upper) / 2
        #         self.class_centers.append(round(center, 2))
        # # ---------------------------------------------------------------------------

        self.normalize = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.fft = args.fft
        self.fft_log_base = args.fft_log_base

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        # class_id = self.class_id[index]

        pil_image = Image.open(image_path).convert('RGB')
        # pil_image = T.functional.resize(pil_image, self.shapes[index])
        image = self.normalize(pil_image)

        if self.fft:
            image = tensor_fft_3D(image, self.fft_log_base)
        
        # if isinstance(image, tuple):
        #     image = torch.stack(image, dim=0)
        return image, self.class_id[index], self.heights[index], image_path

    def __len__(self):
        return len(self.images_paths)
