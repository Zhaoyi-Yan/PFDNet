import random
import math
import os
import numpy as np
import sys
from PIL import Image
from utils import show
from metrics import AEBatch, SEBatch
import time
import torch
import scipy.io as scio

class Estimator(object):
    def __init__(self, setting, eval_loader, criterion=torch.nn.MSELoss(reduction="sum")):
        self.setting = setting
        self.ae_batch = AEBatch().to(self.setting.device)
        self.se_batch = SEBatch().to(self.setting.device)
        self.criterion = criterion
        self.eval_loader = eval_loader
        
    def evaluate(self, model):
        net = model.eval()
        MAE_, MSE_, loss_ = [], [], []
        time_cost = 0
        for eval_img_path, eval_img, eval_gt, eval_pers in self.eval_loader:
            eval_img_path = eval_img_path[0]
            eval_img = eval_img.to(self.setting.device)
            eval_gt = eval_gt.to(self.setting.device)

            start = time.time()
            with torch.no_grad():
                # test cropped patches
                if self.setting.mode == 'crop': 
                    eval_patchs, eval_pers = torch.squeeze(eval_img), torch.squeeze(eval_pers, dim=0)
                    eval_prediction = net(eval_patchs, eval_pers)
                    prediction_map = torch.zeros(eval_gt.shape).to(self.setting.device)
                    self.test_crops(eval_prediction.shape, eval_prediction, prediction_map)
                # test whole images
                elif self.setting.mode == 'whole': 
                    prediction_map = net(eval_img, eval_pers)
                gt_counts = self.get_gt_num(self.setting.eval_gt_path, eval_img_path)
                # calculate metrics
                batch_ae = self.ae_batch(prediction_map, gt_counts).data.cpu().numpy()
                batch_se = self.se_batch(prediction_map, gt_counts).data.cpu().numpy()
                loss = self.criterion(prediction_map, eval_gt)
                loss_.append(loss.data.item())
                MAE_.append(batch_ae)
                MSE_.append(batch_se)
            torch.cuda.synchronize()
            end = time.time()
            time_cost += (end - start)

        # return the validate loss, validate MAE and validate RMSE
        MAE_, MSE_, loss_ = np.reshape(MAE_, [-1]), np.reshape(MSE_, [-1]), np.reshape(loss_, [-1])
        return np.mean(MAE_), np.sqrt(np.mean(MSE_)), np.mean(loss_), time_cost

    def get_gt_num(self, eval_gt_path, img_path):
        tmp_mat_name = os.path.basename(img_path).replace('IMG_', 'GT_IMG_').replace('.jpg', '.mat')
        gt_path = os.path.join(eval_gt_path, os.path.basename(tmp_mat_name))
        gt_counts = len(scio.loadmat(gt_path)['image_info'][0][0][0][0][0])
        return gt_counts
    
    def test_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (i - 1)
                pred_w = math.floor(3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,start_h:start_h + valid_h, start_w:start_w + valid_w]
