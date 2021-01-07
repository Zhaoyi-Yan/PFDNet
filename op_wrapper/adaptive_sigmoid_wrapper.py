import torch
import torch.nn as nn
from torch.autograd import Function
import adaptive_sigmoid_gpu as adaptive_sigmoid

class AdaptiveSigmoidFunction(Function):
    @staticmethod
    def forward(ctx, *args):
        if len(args) != 3:
            print("wrong input parameters number, check the input")
            return
        input = args[0]
        params = args[1]
        ctx.updates_signal = args[2]
        output = adaptive_sigmoid.forward(input, params)
        ctx.save_for_backward(input, params)
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if len(grad_outputs) != 1:
            print("Wrong output number, check your output")
            return
        input, params = ctx.saved_tensors
        grad_copy = grad_outputs[0].clone()
        grad_input, grad_weight= adaptive_sigmoid.backward(input, params, grad_copy, *ctx.updates_signal)
        return grad_input, grad_weight, None
    
class AdaptiveSigmoid(nn.Module):
    def __init__(self, **kwargs):
        super(AdaptiveSigmoid, self).__init__()
        self.params = nn.Parameter(torch.FloatTensor(kwargs['sigma']))
        self.updates_signal = kwargs['updates_signal']
#         self.params.register_hook(print)
        
    def forward(self, x):
        return AdaptiveSigmoidFunction.apply(x, self.params, self.updates_signal)