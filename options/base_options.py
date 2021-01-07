import argparse
import os
import torch
import util.utils as util

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataset_name', default='SHA', help='SHA|SHB|QNRF')
        parser.add_argument('--test_model_name', default='', help='path of pretrained model')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--net_name', type=str, default='csrpersp', help='csrnet|csrpersp')
        parser.add_argument('--mode', type=str, default='whole', help='whole|crop')
        parser.add_argument('--prefix_path', type=str, default='./data', help='path of the dataset folder')
        parser.add_argument('--name', type=str, default='Csrnet_persp', help='name of the experiment.s')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--alpha', type=float, default=1, help='alpha in adaptive sigmoid')
        parser.add_argument('--beta', type=float, default=1, help='beta in adaptive sigmoid')
        parser.add_argument('--gamma', type=float, default=1, help='gamma in adaptive sigmoid')
        parser.add_argument('--theta', type=float, default=2, help='theta in adaptive sigmoid')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./output', help='models are saved here')
        self.initialized = True
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name, opt.dataset_name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)
        opt.isTrain = self.isTrain   # train or test


        self.print_options(opt)

        # set gpu ids
        os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu_ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        # re-order gpu ids
        opt.gpu_ids = [i.item() for i in torch.arange(len(opt.gpu_ids))]
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
