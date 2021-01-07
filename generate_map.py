import cv2
import numpy as np
import scipy
import scipy.io as scio
from PIL import Image
import time
import math
import os
import h5py

def get_density_map_gaussian(H, W, ratio_h, ratio_w,  points, adaptive_kernel=False, fixed_value=15):
    h = H
    w = W
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
        sigma = fixed_value
        sigma = max(1, sigma)

        gaussian_radius = 7
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map

def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


# SHA: 300, 182
# SHB: 400, 316
if __name__ == "__main__":

    is_train = 0 # 0 for test
    train_test = 'train' if is_train else 'test'
    dataset = 'SHA'

    if dataset == 'SHA':
        num_img = 300 if is_train else 182
        image_dir_path = "ShanghaiTech/part_A_final/"+train_test+"_data/images"
        ground_truth_dir_path = "ShanghaiTech/part_A_final/"+train_test+"_data/ground_truth"
        output_gt_dir = "./SH_part_A/"+train_test
    elif dataset == 'SHB':
        num_img = 400 if is_train else 316
        image_dir_path = "ShanghaiTech/part_B_final/"+train_test+"_data/images"
        ground_truth_dir_path = "ShanghaiTech/part_B_final/"+train_test+"_data/ground_truth"
        output_gt_dir = "./SH_part_B/" + train_test
    elif dataset == 'QNRF':
        num_img = 1201 if is_train else 334
        image_dir_path = "UCF-QNRF_ECCV18/" + train_test
        ground_truth_dir_path = "UCF-QNRF_ECCV18/" + train_test
        output_gt_dir = "./QNRF/" + train_test
    elif dataset == 'UCF50': # take all images as testing images
        num_img = 50
        image_dir_path = "UCF_CC_50/images/UCF_CC_50_img"
        ground_truth_dir_path = "UCF_CC_50/UCF_CC_50_mat"
        output_gt_dir = "./UCF50/" + train_test

    mkdirs(output_gt_dir)

    for i in range(num_img):
        if dataset == 'SHA' or dataset == 'SHB':
            img_path = image_dir_path + "/IMG_" + str(i + 1) + ".jpg"
            gt_path = ground_truth_dir_path + "/GT_IMG_" + str(i + 1) + ".mat"
        elif dataset == 'QNRF':
            img_path = os.path.join(image_dir_path, "img_"+("%04d" % (i+1))+".jpg")
            gt_path = os.path.join(image_dir_path, "img_"+("%04d" % (i+1))+"_ann.mat")
        elif dataset == 'UCF50':
            img_path = os.path.join(image_dir_path, ("%d" % (i+1))+".jpg")
            gt_path = os.path.join(ground_truth_dir_path, ("%d" % (i+1))+"_ann.mat")

        img = Image.open(img_path)
        height = img.size[1]
        width = img.size[0]

        if dataset == 'SHA' or dataset == 'SHB':
            points = scio.loadmat(gt_path)['image_info'][0][0][0][0][0]
        elif dataset == 'QNRF':
            points = scio.loadmat(gt_path)['annPoints']
        elif dataset == 'UCF50':
            points = h5py.File(gt_path, 'r')['annPoints'].value.astype(np.float32)


        resize_height = height
        resize_width = width

        if dataset == 'SHA' or dataset == 'UCF50':
            if resize_height <= 416:
                tmp = resize_height
                resize_height = 416
                resize_width = (resize_height / tmp) * resize_width

            if resize_width <= 416:
                tmp = resize_width
                resize_width = 416
                resize_height = (resize_width / tmp) * resize_height

            resize_height = math.ceil(resize_height / 32) * 32
            resize_width = math.ceil(resize_width / 32) * 32
        elif dataset == 'QNRF':
            pass


        ratio_h = (resize_height) / (height)
        ratio_w = (resize_width) / (width)
        # print(height, width, ratio_h, ratio_w)
        gt = get_density_map_gaussian(resize_height, resize_width, ratio_h, ratio_w, points, False, 4)
        gt = np.reshape(gt, [resize_height, resize_width])  # transpose into w, h
        np.save(output_gt_dir + "/GT_IMG_" + str(i + 1), gt)
    print("complete!")
