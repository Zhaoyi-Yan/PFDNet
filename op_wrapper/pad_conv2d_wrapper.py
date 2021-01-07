import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import Module
import pad_conv2d_gpu as pad_conv2d
from op_wrapper.adaptive_sigmoid_wrapper import AdaptiveSigmoid


class PerspectiveDilatedConv2dFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 6:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        weights = args[1]
        rate_map = args[2]
        bias = args[3]
        ctx.stride_h = args[4]
        ctx.stride_w = args[5]
        output = pad_conv2d.forward(input, weights, rate_map, bias, ctx.stride_h, ctx.stride_w)
        ctx.save_for_backward(input, weights, rate_map, bias)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, weights, rate_map, bias = ctx.saved_tensors
        grad_copy = grad_outputs[0].clone()
        grad_input, grad_weight, grad_rate_map, grad_bias = pad_conv2d.backward(input, weights, rate_map, bias, grad_copy, ctx.stride_h, ctx.stride_w)
        return grad_input, grad_weight, grad_rate_map, grad_bias, None, None


class PerspectiveDilatedConv2dLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride_h, stride_w):
        super(PerspectiveDilatedConv2dLayer, self).__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_channels, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight, gain=1)

    def forward(self, inputs, rate_map):
        return PerspectiveDilatedConv2dFunction.apply(inputs, self.weight, rate_map, self.bias, self.stride_h, self.stride_w)

    
class BasicPerspectiveDilatedConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(BasicPerspectiveDilatedConv2D, self).__init__()
        self.rate_map_generator = AdaptiveSigmoid(**kwargs)
#         self.rate_map_generator.params.register_hook(lambda x:print('Conv', x))
        self.stride = 1
        self.pad = (kernel_size // 2)
        self.perspective_dilated_conv2d = PerspectiveDilatedConv2dLayer(in_channels, out_channels, kernel_size, self.stride, self.stride)
        
    def forward(self, x, perspective):
        rate_map = self.rate_map_generator(perspective)
#         rate_map = self.rate_map_generator(x)
        x = torch.nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad ])
        return self.perspective_dilated_conv2d(x, rate_map)

class BasicPerspectiveDilatedConv2D_BN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, **kwargs):
        super(BasicPerspectiveDilatedConv2D_BN, self).__init__()
        self.rate_map_generator = AdaptiveSigmoid(**kwargs)
#         self.rate_map_generator.params.register_hook(lambda x:print('Conv', x))
        self.stride = 1
        self.pad = (kernel_size // 2)
        self.perspective_dilated_conv2d = PerspectiveDilatedConv2dLayer(in_channels, out_channels, kernel_size, self.stride, self.stride)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, perspective):
        rate_map = self.rate_map_generator(perspective)
#         rate_map = self.rate_map_generator(x)
        x = torch.nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad ])
        x = self.perspective_dilated_conv2d(x, rate_map)
        x = self.bn(x)
        return x

