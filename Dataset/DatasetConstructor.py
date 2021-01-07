from PIL import Image
import numpy as np
import os
import glob
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.nn.functional as functional
import torch.utils.data as data
import random
import time
import scipy.io as scio
import h5py
import math

class DatasetConstructor(data.Dataset):
    def __init__(self):
        return
    
    def get_path_tuple(self, i, dataset_name = "SHA", is_pers=True):
        if dataset_name == "SHA" or dataset_name == "SHB":
            img_name = '/IMG_' + str(i + 1) + ".jpg"
            gt_map_name = '/GT_IMG_' + str(i + 1) + ".npy"
            perspective_map_name = ""
            if is_pers:
                perspective_map_name = '/IMG_' + str(i + 1) + ".mat"
        else:
            raise NameError("Only SHA is released")
        return img_name, gt_map_name, perspective_map_name
    
    def resize(self, img, dataset_name):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if dataset_name == "SHA":
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
        else:
            raise NameError("Only SHA is released")
        img = transforms.Resize([resize_height, resize_width])(img)
        return img


class EvalDatasetConstructor(DatasetConstructor):
    def __init__(self,
                 validate_num,
                 data_dir_path,
                 gt_dir_path,
                 pers_dir_path=None,
                 mode="crop",
                 dataset_name="SHA",
                 device=None,
                 ):
        super(EvalDatasetConstructor, self).__init__()
        self.validate_num = validate_num
        self.imgs = []
        self.data_root = data_dir_path
        self.gt_root = gt_dir_path
        self.pers_root = pers_dir_path
        self.mode = mode
        self.device = device
        self.dataset_name = dataset_name
        self.kernel = torch.ones(1, 1, 8, 8, dtype=torch.float32)
        self.kernel_crop = torch.ones(1, 1, 2, 2, dtype=torch.float32)
        self.img_paths = glob.glob(os.path.join(self.data_root, "*.jpg"))

    def __getitem__(self, index):
        if self.mode == 'crop':
            img_path = self.img_paths[index]
            gt_map_path = os.path.join(self.gt_root, os.path.basename(img_path.replace('IMG_', "GT_IMG_"))[:-4]+".npy")
            pers_path = os.path.join(self.pers_root, os.path.basename(img_path.replace('jpg', "mat")))
            img = Image.open(img_path).convert("RGB")
            p_m = np.zeros(img.size[::-1], dtype=float) if self.pers_root == "" else (h5py.File(pers_path, 'r')['pmap'][:] / 100).T
            p_m = super(EvalDatasetConstructor, self).resize(Image.fromarray(p_m), self.dataset_name)
            img = super(EvalDatasetConstructor, self).resize(img, self.dataset_name)
            img = transforms.ToTensor()(img)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))
            gt_map = transforms.ToTensor()(gt_map)
            p_m = transforms.ToTensor()(p_m)
            img_shape, gt_shape = img.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
            imgs, pers = [], []
            for i in range(3):
                for j in range(3):
                    start_h, start_w = (patch_height // 2) * i, (patch_width // 2) * j
                    imgs.append(img[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
                    pers.append(p_m[:, start_h:start_h + patch_height, start_w:start_w + patch_width])
            imgs, pers = torch.stack(imgs), torch.stack(pers)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel_crop, bias=None, stride=2, padding=0)
            return img_path, imgs, gt_map.view(1, gt_shape[1] // 2, gt_shape[2] // 2), pers
        
        elif self.mode == 'whole':
            img_path, gt_map_path, pers_path, img_index = self.imgs[index]
            img = Image.open(img_path).convert("RGB")
            p_m = np.zeros(img.size[::-1], dtype=float) if self.pers_root == "" else (h5py.File(pers_path)['pmap'][:] / 100).T
            p_m = super(EvalDatasetConstructor, self).resize(Image.fromarray(p_m), self.dataset_name)
            img = super(EvalDatasetConstructor, self).resize(img, self.dataset_name)
            img = transforms.ToTensor()(img)
            gt_map = Image.fromarray(np.squeeze(np.load(gt_map_path)))
            gt_map = transforms.ToTensor()(gt_map)
            p_m = transforms.ToTensor()(p_m)
            img_shape, gt_shape = img.shape, gt_map.shape  # C, H, W
            img = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
            gt_map = functional.conv2d(gt_map.view(1, *(gt_shape)), self.kernel, bias=None, stride=8, padding=0)
            return img_path, img, gt_map.view(1, gt_shape[1] // 8, gt_shape[2] // 8), p_m

    def __len__(self):
        return self.validate_num
