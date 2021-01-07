import torch
import torch.nn as nn
from torch.nn import init
import functools
from net.CSRPersNet_crop import CSRPersNet_BN
from net.CSRNet import CSRNet


def init_net(net, init_type='normal', init_gain=0.01, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # Has been initlized inside
    return net

def define_net(opt):
    net_name = opt.net_name
    if net_name == 'csrnet':
        net = CSRNet()
    elif net_name == 'csrpersp_crop':
        net = CSRPersNet_BN(load_path=None,
                        updates_signal=[True, True, True, True], is_relu=False,
                        sigma=[opt.alpha, opt.beta, opt.gamma, opt.theta])
    else:
        raise NotImplementedError('Unrecognized model: '+net_name)
    return net
