import os
import torch
import util.utils as util

class config(object):
    def __init__(self, opt):
        self.opt = opt
        self.min_mae = 10240000
        self.min_loss = 10240000
        self.dataset_name = opt.dataset_name
        self.batch_size = opt.batch_size
        self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.model_save_path = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name) # path of saving model
        self.mode = opt.mode
        prefix_path = opt.prefix_path # prefix path of training path
        if self.dataset_name == "SHA":
            self.eval_num = 182
            self.train_num = 300
            
            self.train_gt_map_path = prefix_path + "/part_A_final/train_data/gt_map_sigma=4_k=7"
            self.train_img_path = prefix_path + "/part_A_final/train_data/images"
            self.train_pers_path = prefix_path + "/part_A_final/train_data/perspective_gt"
            self.eval_gt_map_path = prefix_path + "/part_A_final/test_data/gt_map_sigma=4_k=7"
            self.eval_img_path = prefix_path + "/part_A_final/test_data/images"
            self.eval_gt_path = prefix_path + "/part_A_final/test_data/ground_truth"
            self.eval_pers_path = prefix_path + "/part_A_final/test_data/perspective_gt"
            
        else:
            raise NameError("Only SHA is released currently")
