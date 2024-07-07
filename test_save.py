import argparse
import logging
import os
import random
import numpy as np
import json
import cv2
from PIL import Image
import SimpleITK as sitk
from torchvision import transforms
import matplotlib.pyplot as plt
from logHelper import setup_logger
from config import *
from scipy.spatial import cKDTree
import shutil

args = parse_args()

def calculate_dice(pred, gt):
    # 用于计算 Dice 相似系数
    intersection = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum()

    if union == 0:
        return 1.0  # 如果两者都是空集，Dice 应为 1

    return 2 * intersection / union


def calculate_asd(pred, gt):
    pred_border = np.logical_xor(pred, np.roll(pred, 1, axis=0))
    gt_border = np.logical_xor(gt, np.roll(gt, 1, axis=0))

    pred_border_indices = np.argwhere(pred_border)
    gt_border_indices = np.argwhere(gt_border)

    if len(pred_border_indices) == 0 or len(gt_border_indices) == 0:
        return 0.0  # 无法计算表面距离

    tree_gt = cKDTree(gt_border_indices)
    tree_pred = cKDTree(pred_border_indices)

    distances_to_gt = tree_gt.query(pred_border_indices)[0]
    distances_to_pred = tree_pred.query(gt_border_indices)[0]

    asd = np.mean(distances_to_gt) + np.mean(distances_to_pred)
    asd /= 2.0

    return asd


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1

    if pred.sum() > 0 and gt.sum() > 0:
        dice = calculate_dice(pred, gt)
        asd = calculate_asd(pred, gt)
        return [dice, asd]
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0




if __name__ == "__main__":
    
    data_json_file = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'dataset.json')
    with open(data_json_file) as f:
        json_data = json.load(f)
        num_classes = json_data['label_class_num']
        in_channels = json_data['img_channel']

    input_folder = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'imagesTs')
    output_folder = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'test_pred')
    visualization_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'visualization_result')
    visualization_path_GT = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], 'Ground_Truth', 'visualization_result')
    mask_save_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'mask_result')
    mask_save_path_GT = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], 'Ground_Truth', 'mask_result')


    os.makedirs(visualization_path, exist_ok=True)
    os.makedirs(visualization_path_GT, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)
    os.makedirs(mask_save_path_GT, exist_ok=True)


    test_img_list = os.listdir(input_folder)
    metric_list = []

    for test_img_name in test_img_list:
        img_path = os.path.join(input_folder, test_img_name)
        ground_truth_path = os.path.join(os.environ['medseg_raw'], os.environ['current_dataset'], 'labelsTs', test_img_name)
        prediction_path = os.path.join(output_folder, test_img_name)
        file_ext = test_img_name.split('.')[-1]

        if file_ext in ['png']:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = np.array(img)
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            pred = Image.open(prediction_path)
            pred = np.array(pred)
            ground_truth = Image.open(ground_truth_path)
            ground_truth = np.array(ground_truth)

        elif file_ext in ['gz', 'nrrd', 'mha', 'nii']:
            img = sitk.ReadImage(img_path)
            img = sitk.GetArrayFromImage(img)
            if len(img.shape) == 2:
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
            pred = sitk.ReadImage(prediction_path)
            pred = sitk.GetArrayFromImage(pred)
            ground_truth = sitk.ReadImage(ground_truth_path)
            ground_truth = sitk.GetArrayFromImage(ground_truth)



        each_metric = []
        
        img_with_GT = img.copy()
        for i in range(1, num_classes):
            each_metric.append(calculate_metric_percase(pred == i, ground_truth == i))
            mask = np.where(pred == i, 1, 0).astype(np.uint8)
            colored_mask = np.zeros_like(img)
            colored_mask[mask == 1] = colors[i - 1]
            img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

            gt_mask = np.where(ground_truth == i, 1, 0).astype(np.uint8)
            colored_mask_gt = np.zeros_like(img_with_GT)
            colored_mask_gt[gt_mask == 1] = colors[i - 1]
            img_with_GT = cv2.addWeighted(img_with_GT, 1, colored_mask_gt, 0.5, 0)


        metric_list.append(each_metric)

        visualization_path_each = os.path.join(visualization_path, test_img_name)
        cv2.imwrite(visualization_path_each, img)

        img_with_GT_save_path = os.path.join(visualization_path_GT, test_img_name)
        cv2.imwrite(img_with_GT_save_path, img_with_GT)

        mask_save_path_each = os.path.join(mask_save_path, test_img_name)
        cv2.imwrite(mask_save_path_each, colored_mask)

        mask_save_path_each_GT = os.path.join(mask_save_path_GT, test_img_name)
        cv2.imwrite(mask_save_path_each_GT, colored_mask_gt)

    metric_list = np.array(metric_list)
    print(metric_list.shape)
    dice_each_case = np.mean(metric_list[:, :, 0], axis=1)
    asd_each_case = np.mean(metric_list[:, :, 1], axis=1)

    dice_mean = np.mean(dice_each_case, axis=0)
    dice_std = np.std(dice_each_case, axis=0)

    asd_mean = np.mean(asd_each_case, axis=0)
    asd_std = np.std(asd_each_case, axis=0)

    # save into csv
    print("saving into csv")
    mean_csv_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'test_result_mean.csv')
    with open(mean_csv_path, 'w') as f:
        f.write('dice_mean,dice_std,asd_mean,asd_std\n')
        f.write(f'{dice_mean},{dice_std},{asd_mean},{asd_std}\n')

    print("saving into csv")
    csv_path = os.path.join(os.environ['medseg_results'], os.environ['current_dataset'], os.environ['MODEL_NAME'], 'test_result.csv')
    with open(csv_path, 'w') as f:
        f.write('dice,asd\n')
        for each_metric in metric_list:
            f.write(f'{each_metric[0][0]},{each_metric[0][1]}\n')
    print("saving into csv, finished")

    shutil.copy(visualization_path_each, 'static/result_visiual.png')
    print(f'DICE: {dice_mean} ± {dice_std}')
    with open(output_file, 'a') as f:
        f.write(f'\nimg_with_mask_save_path: {visualization_path}\n')
        f.write(f'\nmask_save_path: {mask_save_path}\n')
        f.write(f'\nresult_csv_path: {csv_path}')
        f.write(f'\nDICE: {dice_mean} ± {dice_std}\nASD: {asd_mean} ± {asd_std}')

