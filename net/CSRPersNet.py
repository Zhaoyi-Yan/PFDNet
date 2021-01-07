import torch
import torch.nn as nn
from torchvision import models
from op_wrapper.pad_conv2d_wrapper import BasicPerspectiveDilatedConv2D
from op_wrapper.adaptive_sigmoid_wrapper import AdaptiveSigmoid
from op_wrapper.pad_conv2d_wrapper import PerspectiveDilatedConv2dLayer
from collections import OrderedDict
pretrain_dict = nn.ModuleList(list(list(models.vgg16(True).children())[0].children())[0:23]).state_dict()

class Frontend(nn.Module):
    def __init__(self, pretrain=True, **kwargs):
        super(Frontend, self).__init__()
        self.pad_conv2d_1 = BasicPerspectiveDilatedConv2D(3, 64, 3, 1, **kwargs)
        self.pad_relu_1 = nn.ReLU(inplace=True)
        self.pad_conv2d_2 = BasicPerspectiveDilatedConv2D(64, 64, 3, 1, **kwargs)
        self.pad_relu_2 = nn.ReLU(inplace=True)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
        
        self.pad_conv2d_3 = BasicPerspectiveDilatedConv2D(64, 128, 3, 1, **kwargs)
        self.pad_relu_3 = nn.ReLU(inplace=True)
        self.pad_conv2d_4 = BasicPerspectiveDilatedConv2D(128, 128, 3, 1, **kwargs)
        self.pad_relu_4 = nn.ReLU(inplace=True)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
        
        self.pad_conv2d_5 = BasicPerspectiveDilatedConv2D(128, 256, 3, 1, **kwargs)
        self.pad_relu_5 = nn.ReLU(inplace=True)
        self.pad_conv2d_6 = BasicPerspectiveDilatedConv2D(256, 256, 3, 1, **kwargs)
        self.pad_relu_6 = nn.ReLU(inplace=True)
        self.pad_conv2d_7 = BasicPerspectiveDilatedConv2D(256, 256, 3, 1, **kwargs)
        self.pad_relu_7 = nn.ReLU(inplace=True)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)
        
        self.pad_conv2d_8 = BasicPerspectiveDilatedConv2D(256, 512, 3, 1, **kwargs)
        self.pad_relu_8 = nn.ReLU(inplace=True)
        self.pad_conv2d_9 = BasicPerspectiveDilatedConv2D(512, 512, 3, 1, **kwargs)
        self.pad_relu_9 = nn.ReLU(inplace=True)
        self.pad_conv2d_10 = BasicPerspectiveDilatedConv2D(512, 512, 3, 1, **kwargs)
        self.pad_relu_10 = nn.ReLU(inplace=True)
        if pretrain == True:
            new_state_dict = OrderedDict()
            count = 1
            for k,v in pretrain_dict.items():
                name_prefix = "pad_conv2d_" + str(count)
                if 'weight' in k:
                    new_state_dict[name_prefix + '.rate_map_generator.params'] = torch.FloatTensor(kwargs['sigma'])
                    new_state_dict[name_prefix + '.perspective_dilated_conv2d.weight'] = v
                elif 'bias' in k:
                    new_state_dict[name_prefix + '.perspective_dilated_conv2d.bias'] = v
                    count += 1
            self.load_state_dict(new_state_dict)
        
    
    def forward(self, x, perspective_map):
        x = self.pad_conv2d_1(x, perspective_map)
        x = self.pad_relu_1(x)
        x = self.pad_conv2d_2(x, perspective_map)
        x = self.pad_relu_2(x)
        x = self.max_pool_1(x)
        perspective_map = self.max_pool_1(perspective_map)
        
        x = self.pad_conv2d_3(x, perspective_map)
        x = self.pad_relu_3(x)
        x = self.pad_conv2d_4(x, perspective_map)
        x = self.pad_relu_4(x)
        x = self.max_pool_2(x)
        perspective_map = self.max_pool_2(perspective_map)
        
        x = self.pad_conv2d_5(x, perspective_map)
        x = self.pad_relu_5(x)
        x = self.pad_conv2d_6(x, perspective_map)
        x = self.pad_relu_6(x)
        x = self.pad_conv2d_7(x, perspective_map)
        x = self.pad_relu_7(x)
        x = self.max_pool_3(x)
        perspective_map = self.max_pool_3(perspective_map)
        
        x = self.pad_conv2d_8(x, perspective_map)
        x = self.pad_relu_8(x)
        x = self.pad_conv2d_9(x, perspective_map)
        x = self.pad_relu_9(x)
        x = self.pad_conv2d_10(x, perspective_map)
        x = self.pad_relu_10(x)
        
        return x, perspective_map

class Backend(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(Backend, self).__init__()
        self.pad_conv2d_1 = BasicPerspectiveDilatedConv2D(in_channels, 512, 3, 1,  **kwargs)
        self.pad_relu_1 = nn.ReLU(inplace=True)
        self.pad_conv2d_2 = BasicPerspectiveDilatedConv2D(512, 512, 3, 1, **kwargs)
        self.pad_relu_2 = nn.ReLU(inplace=True)
        self.pad_conv2d_3 = BasicPerspectiveDilatedConv2D(512, 512, 3, 1, **kwargs)
        self.pad_relu_3 = nn.ReLU(inplace=True)
        self.pad_conv2d_4 = BasicPerspectiveDilatedConv2D(512, 256, 3, 1, **kwargs)
        self.pad_relu_4 = nn.ReLU(inplace=True)
        self.pad_conv2d_5 = BasicPerspectiveDilatedConv2D(256, 128, 3, 1, **kwargs)
        self.pad_relu_5 = nn.ReLU(inplace=True)
        self.pad_conv2d_6 = BasicPerspectiveDilatedConv2D(128, 64, 3, 1, **kwargs)
        self.pad_relu_6 = nn.ReLU(inplace=True)
    
    def forward(self, x, perspective_map):
        x = self.pad_conv2d_1(x, perspective_map)
        x = self.pad_relu_1(x)
        x = self.pad_conv2d_2(x, perspective_map)
        x = self.pad_relu_2(x)
        x = self.pad_conv2d_3(x, perspective_map)
        x = self.pad_relu_3(x)
        x = self.pad_conv2d_4(x, perspective_map)
        x = self.pad_relu_4(x)
        x = self.pad_conv2d_5(x, perspective_map)
        x = self.pad_relu_5(x)
        x = self.pad_conv2d_6(x, perspective_map)
        x = self.pad_relu_6(x)
        return x

class CSRPersNet(nn.Module):
    def __init__(self, load_path=None, **kwargs):
        super(CSRPersNet, self).__init__()
        self.front_end = Frontend(True, **kwargs)
        self.back_end = Backend(512, **kwargs)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not(load_path == None):
            new_state_dict = OrderedDict()
            state_dict = torch.load(load_path)
            count = 1
            for k,v in state_dict.items():
                if 'back_end' in k:
                    name_prefix = "back_end.pad_conv2d_" + str(count)
                    if 'weight' in k:
                        new_state_dict[name_prefix + '.rate_map_generator.params'] = torch.FloatTensor(*kwargs)
                        new_state_dict[name_prefix + '.perspective_dilated_conv2d.weight'] = v
                    elif 'bias' in k:
                        new_state_dict[name_prefix + '.perspective_dilated_conv2d.bias'] = v
                        count += 1
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict)
            
        else:
            for m in self.output_layer.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                
    def forward(self, x, perspective_map):
        x, perspective_map = self.front_end(x, perspective_map)
        x = self.back_end(x, perspective_map)
        x = self.output_layer(x)
        return x

    def get_params(self):
        self.ada_sig_params = []
        self.conv_params = []
        for m in self.modules():
            if isinstance(m, AdaptiveSigmoid):
                self.ada_sig_params.append(m.params)
            elif isinstance(m, nn.Conv2d):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)
            elif isinstance(m, PerspectiveDilatedConv2dLayer):
                self.conv_params.append(m.weight)
                self.conv_params.append(m.bias)
        return self.conv_params, self.ada_sig_params