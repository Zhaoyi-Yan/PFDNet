import torch
import torch.nn as nn
from torch.nn import init
import functools
from net.CSRPersNet_crop import CSRPersNet_BN
from net.CSRNet import CSRNet


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

def select_optim(net, opt):
    if opt.optimizer == 'adam':
        if opt.net_name == "csrnet":
            optimizer = torch.optim.Adam(net.parameters(), opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
            return optimizer
        else: # csrpersp
            main_params, main_bn_params, ada_sig_params = net.module.get_params()
            optimizer_base = torch.optim.Adam({*main_params}, opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
            optimizer_base_bn = torch.optim.Adam({*main_bn_params}, opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
            optimizer_sig = torch.optim.Adam({*ada_sig_params}, opt.sig_times*opt.lr, weight_decay=opt.weight_decay, amsgrad=opt.amsgrad)
            return optimizer_base, optimizer_base_bn, optimizer_sig
    else:
        raise NotImplementedError('This optimizer has not implemented yet')



def init_net(net, init_type='normal', init_gain=0.01, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    # later using custom init, for now, just init inside.
    # init_weights(net, init_type, init_gain=init_gain)
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
