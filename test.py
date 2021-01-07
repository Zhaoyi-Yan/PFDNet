# config
import sys
import numpy as np
import torch
from config import config
import warnings
import time
from options.test_options import TestOptions
from Dataset.DatasetConstructor import EvalDatasetConstructor
from eval.Estimator import Estimator
from net.CSRPersNet import CSRPersNet
from net.CSRNet import CSRNet
import net.networks as networks

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batch_size = 1  # test code only supports batchSize = 1
opt.is_flip = 0  # no flip

# Mainly get settings for specific datasets
setting = config(opt)


eval_dataset = EvalDatasetConstructor(
    setting.eval_num,
    setting.eval_img_path,
    setting.eval_gt_map_path,
    setting.eval_pers_path,
    mode=setting.mode,
    dataset_name=setting.dataset_name,
    device=setting.device)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=1)

# model construct
net = networks.define_net(opt)
net = networks.init_net(net, gpu_ids=opt.gpu_ids)

net.module.load_state_dict(torch.load(opt.test_model_name, map_location=str(setting.device)))
criterion = torch.nn.MSELoss(reduction='sum').to(setting.device)
estimator = Estimator(setting, eval_loader, criterion=criterion)

# False means do not choose a random sample to show
validate_MAE, validate_RMSE, validate_loss, time_cost = estimator.evaluate(net) 
sys.stdout.write('loss = {}, eval_mae = {}, eval_rmse = {}, time cost eval = {}s\n'
                .format(validate_loss, validate_MAE, validate_RMSE, time_cost))
sys.stdout.flush()
