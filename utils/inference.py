import torch
from tqdm import tqdm, trange
import numpy as np

from torch.utils.data import DataLoader
from models.classifiers.classifiers import LinearLayer

import pandas as pd

from utils.pipeline_ops import CLASSES_CENTERS, CLASSES_NUM

LR_N = [1, 2, 3]
threshold_list = [25, 50]

# 评判3个标准：1. 召回的前1个类和前3个类的center height在不在Threshold里；2. 所有的query中分类对了的占总体的多少；3. 召回的第一个类对应的center和真实height的差值，在Threshold里的占所有query的百分比。

def compute_pred(criterion, descriptors):
    if isinstance(criterion, LinearLayer):
        # Using LinearLayer
        return criterion(descriptors, None)[0]
    else:
        # Using AMCC/LMCC
        try:
            return torch.mm(descriptors, criterion.weight.t())
        except:
            return torch.mm(descriptors, criterion.weight)

def inference(args, model:torch.nn.Module, classifier, test_dl:DataLoader, test_img_num, classes_centers=CLASSES_CENTERS):
    # try:
    #    threshold = args.threshold
    # except:
    #     threshold = 30
    if args.threshold is not None:
       threshold = args.threshold
    else:
        threshold = 30
    # TODO 这里我要返回两个量，一个是平均MSE误差（开方），一个是按照各个高度真实值大小排列的误差平均值、最大误差、最小误差、中间值，以后可以画图看不同高度上的结果
    # tqdm_bar = tqdm(range(test_img_num), ncols=100, desc="")
    model = model.to(args.device)
    model.eval()

    classifier = classifier.to(args.device)

    query_heights = torch.zeros(test_img_num)
    query_labels = torch.zeros(test_img_num)
    pred_labels = torch.zeros(test_img_num, CLASSES_NUM)
    pred_heights = torch.zeros(test_img_num, CLASSES_NUM)
    pred_distances = torch.zeros(test_img_num, CLASSES_NUM)
    # global classes_centers
    classes_centers = torch.tensor(classes_centers)
    
    device = args.device
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        tbar = trange(test_img_num, ncols=100)
        for query_i, (image, query_label, query_height) in enumerate(test_dl):
            image = image.to(device)
            query_label = torch.as_tensor(query_label, device=device)
            query_height = torch.as_tensor(query_height, device=device)
            query_labels[query_i] = query_label
            query_heights[query_i] = query_height
    
            descriptor = model(image)

            pred = compute_pred(classifier, descriptor)    # NOTE classifier中的weight包含了12个分类的模式向量

            pred_id_sorted = pred.argsort(descending=True)
            assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
            classes_centers = classes_centers.to(device)
            pred_centers_sorted = classes_centers[pred_id_sorted]

            pred_labels[query_i] = pred_id_sorted
            pred_heights[query_i] = pred_centers_sorted

            pred_centers_length = pred_centers_sorted.shape[1]
            dist = torch.zeros(pred_centers_length)
            for i in range(pred_centers_length):
                dist[i] = pred_centers_sorted[0,i].to(torch.float64) - query_height[0,0]
            dist = torch.abs(dist)
                        
            pred_distances[query_i] = dist
            tbar.update(1)
            
    classifier = classifier.cpu()
    torch.cuda.empty_cache()  # Release classifier memory
    # 正确分类的比例
    pred_1st_labels = pred_labels[:, 0]
    correct_labels = (pred_1st_labels == query_labels)
    lr_1st_correct = torch.count_nonzero(correct_labels).item() * 100 / test_img_num

    # 第一个召回的在若干Threshold范围
    pred_distances_1st = pred_distances[:, 0]
    lr_1st = []
    for thresh in threshold_list:
        lr = torch.count_nonzero((pred_distances_1st <= thresh)).item() * 100 / test_img_num
        lr_1st.append(lr)

    # 前1个和前3个召回的在treshold范围内的比例
    lr_in_thresh = []
    for N in LR_N:
        lr = torch.count_nonzero((pred_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / test_img_num
        lr_in_thresh.append(lr)

    correct_1st_recall = f'the proportion of the top 1 recall is correctly labelled: LR@1: {lr_1st_correct:.1f}'
    in_thresh_1st_recall = ", ".join([f'LR{threshold}m@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_in_thresh)])
    in_thresh_1_3_recall = ", ".join([f'LR{thresh:}m@1: {acc:.1f}' for thresh, acc in zip(threshold_list, lr_1st)])
    
    return correct_1st_recall, in_thresh_1st_recall, in_thresh_1_3_recall


def inference_with_groups(args, model:torch.nn.Module, classifiers, test_dl:DataLoader, groups, num_test_images):
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
    pred_class_ids_collection = torch.zeros(num_test_images, max(LR_N))
    query_class_ids_collection = torch.zeros(num_test_images, 1)


    all_preds_heights_centers = [center for group in groups for center in group.class_centers]
    all_preds_heights_centers = torch.tensor(all_preds_heights_centers).to(args.device)
    all_preds_class_id = [class_id for group in groups for class_id in group.classes_ids]
    all_preds_class_id = torch.tensor(all_preds_class_id).to(args.device)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_class_ids, query_heights, _) in enumerate(tqdm(test_dl, ncols=100)):
            query_class_ids_collection[query_i,:] = query_class_ids
            images = images.to(args.device) # 设置batch_size=1的时候实际上一次循环只有一张图
            query_heights = torch.as_tensor(query_heights, device=args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
                
            # topn_pred_class_id_idx = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            # topn_pred_class_id = all_preds_heights_centers[topn_pred_class_id_idx]
            top_to_low_pred_class_id_idx = all_preds_confidences.argsort(descending=True)
            pred_class_centers = all_preds_heights_centers[top_to_low_pred_class_id_idx]
            pred_class_ids = all_preds_class_id[top_to_low_pred_class_id_idx]

            topn_pred_class_id = pred_class_ids[0:max(LR_N)]
            topn_pred_class_centers = pred_class_centers[0:max(LR_N)]
            
            dist = torch.abs(topn_pred_class_centers.to(torch.float64) - query_heights)
            valid_distances[query_i] = dist

            query_class_ids_collection[query_i] = query_class_ids
            pred_class_ids_collection[query_i] = topn_pred_class_id


    if args.threshold is not None:
       threshold = args.threshold
    else:
        threshold = 30
    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory

    # 先求第一个召回的class和前三个召回的class有没有分对的
    lr_topn_class_right = []
    for N in LR_N:
        correct_labels = torch.zeros_like(pred_class_ids_collection[:,:N], dtype=torch.bool)
        for img_idx in range(num_test_images):
            correct_labels[img_idx] = (pred_class_ids_collection[img_idx,:N] == query_class_ids_collection[img_idx])
        correct_labels_num = correct_labels.any(dim=1)
        correct_percentage = torch.count_nonzero(correct_labels_num).item() * 100 / num_test_images
        lr_topn_class_right.append(correct_percentage)

    correct_classes_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_topn_class_right)])

    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_images)
    
    gcd_str = f"set the threshold to {threshold}m, " + ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_ns)])

    mean_distance = torch.mean(valid_distances).item()
    gcd_str += f", \nmean distance: {mean_distance:.2f}"
    threshold_group = [25, 50, 100]
    for thresh in threshold_group:
        tmp = torch.count_nonzero(valid_distances[:, 0] <= thresh).item() * 100 / num_test_images

        gcd_str += f", \nLR@{thresh}m: {tmp:.2f}"


    
    return correct_classes_str, gcd_str

def inference_with_groups_with_val(args, model:torch.nn.Module, classifiers, test_dl:DataLoader, groups, num_test_images):
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
    pred_class_ids_collection = torch.zeros(num_test_images, max(LR_N))
    query_class_ids_collection = torch.zeros(num_test_images, 1)


    all_preds_heights_centers = [center for group in groups for center in group.class_centers]
    all_preds_heights_centers = torch.tensor(all_preds_heights_centers).to(args.device)
    all_preds_class_id = [class_id for group in groups for class_id in group.classes_ids]
    all_preds_class_id = torch.tensor(all_preds_class_id).to(args.device)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_class_ids, query_heights, _) in enumerate(tqdm(test_dl, ncols=100)):
            query_class_ids_collection[query_i,:] = query_class_ids
            images = images.to(args.device) # 设置batch_size=1的时候实际上一次循环只有一张图
            query_heights = torch.as_tensor(query_heights, device=args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
                
            # topn_pred_class_id_idx = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            # topn_pred_class_id = all_preds_heights_centers[topn_pred_class_id_idx]
            top_to_low_pred_class_id_idx = all_preds_confidences.argsort(descending=True)
            pred_class_centers = all_preds_heights_centers[top_to_low_pred_class_id_idx]
            pred_class_ids = all_preds_class_id[top_to_low_pred_class_id_idx]

            topn_pred_class_id = pred_class_ids[0:max(LR_N)]
            topn_pred_class_centers = pred_class_centers[0:max(LR_N)]
            
            dist = torch.abs(topn_pred_class_centers.to(torch.float64) - query_heights)
            valid_distances[query_i] = dist

            query_class_ids_collection[query_i] = query_class_ids
            pred_class_ids_collection[query_i] = topn_pred_class_id


    if args.threshold is not None:
       threshold = args.threshold
    else:
        threshold = 30
    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory

    # 先求第一个召回的class和前三个召回的class有没有分对的
    lr_topn_class_right = []
    for N in LR_N:
        correct_labels = torch.zeros_like(pred_class_ids_collection[:,:N], dtype=torch.bool)
        for img_idx in range(num_test_images):
            correct_labels[img_idx] = (pred_class_ids_collection[img_idx,:N] == query_class_ids_collection[img_idx])
        correct_labels_num = correct_labels.any(dim=1)
        correct_percentage = torch.count_nonzero(correct_labels_num).item() * 100 / num_test_images
        lr_topn_class_right.append(correct_percentage)

    correct_classes_str = ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_topn_class_right)])

    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_images)

    gcd_str = f"set the threshold to {threshold}m, " + ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_ns)])
    
    return correct_classes_str, gcd_str, lr_ns[0]

def inference_with_groups_csv(args, model:torch.nn.Module, classifiers, test_dl:DataLoader, groups, num_test_images):
    # CSV文件中记录实际高度、推测高度、实际utm，需要结合原始图像信息进行推测；应该返回一下是否推测正确？
    
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = 30
    torch.cuda.empty_cache()  # Release classifiers memory

    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
    pred_class_ids_collection = torch.zeros(num_test_images, max(LR_N))
    query_class_ids_collection = torch.zeros(num_test_images, 1)


    all_preds_heights_centers = [center for group in groups for center in group.class_centers]
    all_preds_heights_centers = torch.tensor(all_preds_heights_centers).to(args.device)
    all_preds_class_id = [class_id for group in groups for class_id in group.classes_ids]
    all_preds_class_id = torch.tensor(all_preds_class_id).to(args.device)

    images_info = []
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_class_ids, query_heights, images_paths) in enumerate(tqdm(test_dl, ncols=100)):
            query_class_ids_collection[query_i,:] = query_class_ids
            images = images.to(args.device) # 设置batch_size=1的时候实际上一次循环只有一张图
            query_heights = torch.as_tensor(query_heights, device=args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
            
                
            # topn_pred_class_id_idx = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            # topn_pred_class_id = all_preds_heights_centers[topn_pred_class_id_idx]
            top_to_low_pred_class_id_idx = all_preds_confidences.argsort(descending=True)
            pred_class_centers = all_preds_heights_centers[top_to_low_pred_class_id_idx]
            pred_class_ids = all_preds_class_id[top_to_low_pred_class_id_idx]


            

            topn_pred_class_id = pred_class_ids[0:max(LR_N)]
            topn_pred_class_centers = pred_class_centers[0:max(LR_N)]
            
            dist = torch.abs(topn_pred_class_centers.to(torch.float64) - query_heights)
            valid_distances[query_i] = dist


            image_info = {'image_path': images_paths[0], 'query_height': query_heights.item(), 'pred_height': pred_class_centers[0].item(), 'in_threshold': (dist[0] <= threshold).item()}
            images_info.append(image_info)

            query_class_ids_collection[query_i] = query_class_ids
            pred_class_ids_collection[query_i] = topn_pred_class_id
    
    return images_info


def inference_sues_with_val(args, model:torch.nn.Module, classifiers, test_dl:DataLoader, groups, num_test_images):
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
    pred_class_ids_collection = torch.zeros(num_test_images, max(LR_N))
    query_class_ids_collection = torch.zeros(num_test_images, 1)


    all_preds_heights_centers = [center for group in groups for center in group.class_centers]
    all_preds_heights_centers = torch.tensor(all_preds_heights_centers).to(args.device)
    all_preds_class_id = [class_id for group in groups for class_id in group.classes_ids]
    all_preds_class_id = torch.tensor(all_preds_class_id).to(args.device)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_class_ids, query_heights, _) in enumerate(tqdm(test_dl, ncols=100)):
            query_class_ids_collection[query_i,:] = query_class_ids
            images = images.to(args.device) # 设置batch_size=1的时候实际上一次循环只有一张图
            query_heights = torch.as_tensor(query_heights, device=args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
                
            # topn_pred_class_id_idx = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            # topn_pred_class_id = all_preds_heights_centers[topn_pred_class_id_idx]
            top_to_low_pred_class_id_idx = all_preds_confidences.argsort(descending=True)
            pred_class_centers = all_preds_heights_centers[top_to_low_pred_class_id_idx]
            pred_class_ids = all_preds_class_id[top_to_low_pred_class_id_idx]

            topn_pred_class_id = pred_class_ids[0:max(LR_N)]
            topn_pred_class_centers = pred_class_centers[0:max(LR_N)]
            
            dist = torch.abs(topn_pred_class_centers.to(torch.float64) - query_heights)
            valid_distances[query_i] = dist

            query_class_ids_collection[query_i] = query_class_ids
            pred_class_ids_collection[query_i] = topn_pred_class_id


    # if args.threshold is not None:
    #    threshold = args.threshold
    # else:
    #     threshold = 30
    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory

    # 先求第一个召回的class和前三个召回的class有没有分对的
    lr_topn_class_right = []
    for N in LR_N:
        correct_labels = torch.zeros_like(pred_class_ids_collection[:,:N], dtype=torch.bool)
        for img_idx in range(num_test_images):
            correct_labels[img_idx] = (pred_class_ids_collection[img_idx,:N] == query_class_ids_collection[img_idx])
        correct_labels_num = correct_labels.any(dim=1)
        correct_percentage = torch.count_nonzero(correct_labels_num).item() * 100 / num_test_images
        lr_topn_class_right.append(correct_percentage)

    correct_classes_str = ", ".join([f'LR@{N}: {acc:.1f}' for N, acc in zip(LR_N, lr_topn_class_right)])

    return correct_classes_str, lr_topn_class_right[0]

import psutil
import time
def inference_latency_memory(args, model:torch.nn.Module, classifiers, test_dl:DataLoader, groups, num_test_images):
     # 初始化性能指标
    latencies = []
    cpu_mem_deltas = []
    cpu_mem_peaks = []
    gpu_mem_peaks = []
    
    # 获取初始内存状态
    process = psutil.Process()
    initial_cpu_mem = process.memory_info().rss / 1024 / 1024  # MB
    
    model = model.eval()
    classifiers = [c.to(args.device) for c in classifiers]
    valid_distances = torch.zeros(num_test_images, max(LR_N))
    pred_class_ids_collection = torch.zeros(num_test_images, max(LR_N))
    query_class_ids_collection = torch.zeros(num_test_images, 1)


    all_preds_heights_centers = [center for group in groups for center in group.class_centers]
    all_preds_heights_centers = torch.tensor(all_preds_heights_centers).to(args.device)
    all_preds_class_id = [class_id for group in groups for class_id in group.classes_ids]
    all_preds_class_id = torch.tensor(all_preds_class_id).to(args.device)
    
    with torch.no_grad():
        # query_class are the UTMs of the center of the class to which the query belongs
        for query_i, (images, query_class_ids, query_heights, _) in enumerate(tqdm(test_dl, ncols=100)):
            # 记录开始时间和内存
            start_time = time.time()
            start_cpu_mem = process.memory_info().rss / 1024 / 1024
            start_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        
            query_class_ids_collection[query_i,:] = query_class_ids
            images = images.to(args.device) # 设置batch_size=1的时候实际上一次循环只有一张图
            query_heights = torch.as_tensor(query_heights, device=args.device)
            descriptors = model(images)
            
            all_preds_confidences = torch.zeros([0], device=args.device)
            for i in range(len(classifiers)):
                pred = compute_pred(classifiers[i], descriptors)
                assert pred.shape[0] == 1  # pred has shape==[1, num_classes]
                all_preds_confidences = torch.cat([all_preds_confidences, pred[0]])
            
                
            # topn_pred_class_id_idx = all_preds_confidences.argsort(descending=True)[:max(LR_N)]
            # topn_pred_class_id = all_preds_heights_centers[topn_pred_class_id_idx]
            top_to_low_pred_class_id_idx = all_preds_confidences.argsort(descending=True)
            pred_class_centers = all_preds_heights_centers[top_to_low_pred_class_id_idx]
            pred_class_ids = all_preds_class_id[top_to_low_pred_class_id_idx]

            # 记录结束时间和内存
            end_time = time.time()
            end_cpu_mem = process.memory_info().rss / 1024 / 1024
            end_gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
            # 计算指标
            latency = (end_time - start_time) * 1000  # 转换为毫秒
            cpu_delta = end_cpu_mem - start_cpu_mem
            cpu_peak = end_cpu_mem - initial_cpu_mem
            gpu_peak = end_gpu_mem
            # 保存指标
            latencies.append(latency)
            cpu_mem_deltas.append(cpu_delta)
            cpu_mem_peaks.append(cpu_peak)
            gpu_mem_peaks.append(gpu_peak)

            topn_pred_class_id = pred_class_ids[0:max(LR_N)]
            topn_pred_class_centers = pred_class_centers[0:max(LR_N)]
            
            dist = torch.abs(topn_pred_class_centers.to(torch.float64) - query_heights)
            valid_distances[query_i] = dist

            query_class_ids_collection[query_i] = query_class_ids
            pred_class_ids_collection[query_i] = topn_pred_class_id
    print(latencies)
    # 计算最终统计结果
    avg_latency = np.mean(latencies)
    max_latency = np.max(latencies)
    avg_cpu_delta = np.mean(cpu_mem_deltas)
    max_cpu_peak = np.max(cpu_mem_peaks)
    max_gpu_peak = np.max(gpu_mem_peaks)
    


    if args.threshold is not None:
       threshold = args.threshold
    else:
        threshold = 30
    classifiers = [c.cpu() for c in classifiers]
    torch.cuda.empty_cache()  # Release classifiers memory

    # 先求第一个召回的class和前三个召回的class有没有分对的
    lr_topn_class_right = []
    for N in LR_N:
        correct_labels = torch.zeros_like(pred_class_ids_collection[:,:N], dtype=torch.bool)
        for img_idx in range(num_test_images):
            correct_labels[img_idx] = (pred_class_ids_collection[img_idx,:N] == query_class_ids_collection[img_idx])
        correct_labels_num = correct_labels.any(dim=1)
        correct_percentage = torch.count_nonzero(correct_labels_num).item() * 100 / num_test_images
        lr_topn_class_right.append(correct_percentage)

    correct_classes_str = ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_topn_class_right)])

    lr_ns = []
    for N in LR_N:
        lr_ns.append(torch.count_nonzero((valid_distances[:, :N] <= threshold).any(axis=1)).item() * 100 / num_test_images)
    
    gcd_str = f"set the threshold to {threshold}m, " + ", ".join([f'LR@{N}: {acc:.2f}' for N, acc in zip(LR_N, lr_ns)])

    mean_distance = torch.mean(valid_distances).item()
    gcd_str += f", \nmean distance: {mean_distance:.2f}"
    threshold_group = [25, 50, 100]
    for thresh in threshold_group:
        tmp = torch.count_nonzero(valid_distances[:, 0] <= thresh).item() * 100 / num_test_images

        gcd_str += f", \nLR@{thresh}m: {tmp:.2f}"

    return avg_latency, max_latency, max_cpu_peak, avg_cpu_delta, max_gpu_peak
