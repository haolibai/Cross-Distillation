# Code from https://github.com/simochen/model-tools.
import numpy as np
import pdb
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random


def print_model_param_nums(model, multiply_adds=True):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of original params: %.8fM' % (total / 1e6))


def print_pruned_param_nums(model, mask_list, multiply_adds=True):
    total = 0
    conv_layer_id = 0
    for m in model.modules():
      try:
        W = m.weight  # has weight
        if isinstance(m, nn.Conv2d) and m.kernel_size != (1,1):
          # TODO: 1x1 conv in resnet not supported
          total += mask_list[conv_layer_id].sum().item()
          conv_layer_id += 1
        elif isinstance(m, nn.BatchNorm2d):
          # TODO: consider running_mean/bias/var?
          mask =  mask_list[conv_layer_id-1][:, :, 0, 0].sum(dim=1) # co
          mask = mask / mask.max()
          total += mask.sum().item()
        else:
          # add linear layers
          total += W.nelement()
      except AttributeError:
        continue

    assert conv_layer_id == len(mask_list)
    print('  + Number of pruned params: %.8fM' % (total / 1e6))


def print_model_param_flops(model, input_res, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size
        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    input = Variable(torch.rand(3, 3, input_res, input_res), requires_grad = True).cuda()
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs of original model: %.8fG' % (total_flops / 3 / 1e9))
    # print('list_conv', list_conv)
    # print('list_linear', list_linear)
    # print('list_bn', list_bn)
    # print('list_relu', list_relu)
    # print('list_pooling', list_pooling)

    return total_flops

def print_pruned_param_flops(model, input_res, mask_list, multiply_adds=True):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_conv=[]
    def conv_hook(self, input, output):

        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        if id(self) in mask_dict_conv:
          mask = mask_dict_conv[id(self)]
          in_sparsity = mask.sum(dim=(0,2,3)).nonzero().squeeze().shape[0] / input_channels
          out_sparsity = mask.sum(dim=(1,2,3)).nonzero().squeeze().shape[0] / output_channels
          # print("mask found in dict:", self, "in sparse: %.2f, out sparse: %.2f" % (in_sparsity, out_sparsity))
        else:
          # print(m, "Not found")
          in_sparsity = 1.
          out_sparsity = 1.

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * ((self.in_channels*in_sparsity) / self.groups)
        bias_ops = 1 if self.bias is not None else 0
        params = output_channels * (kernel_ops + bias_ops) * out_sparsity
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * (output_channels * out_sparsity) * output_height * output_width * batch_size
        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1
        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        if id(self) in mask_dict_bn:
          mask = mask_dict_bn[id(self)]
          sparsity = mask.sum().item() / len(mask)
        else:
          # the bn in the shortcut
          sparsity = 1.
        list_bn.append(input[0].nelement() * sparsity * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        # if input[0].ndimension() == 4:
        #   mask = mask_dict_relu[id(self)]
        #   sparsity = mask.sum().item() / len(mask)
        # else:
        #   sparsity = 1.

        if input[0].ndimension() == 4:
          in_channels = input[0].shape[1]
          sparsity = input[0].sum(dim=(0,2,3)).nonzero().squeeze().shape[0] / in_channels
        else:
          sparsity = 1.
        list_relu.append(input[0].nelement() * sparsity)

    list_pooling=[]
    def pooling_hook(self, input, output):
        # if input[0].ndimension() == 4:
        #   mask = mask_dict_pool[id(self)]
        #   sparsity = mask.sum().item() / len(mask)
        # else:
        #   sparsity = 1.

        in_channels = input[0].shape[1]
        sparsity = input[0].sum(dim=(0,2,3)).nonzero().squeeze().shape[0] / in_channels
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * (output_channels * sparsity) * output_height * output_width * batch_size
        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    # construct mask_dict
    mask_dict_conv = {}
    mask_dict_bn = {}
    # mask_dict_relu = {}
    # mask_dict_pool = {}
    conv_layer_id = 0
    # FIXME: identify the proper modules to add in dicts.
    # FIXME: can only traject flops in modules.
    for m in model.modules():
      if isinstance(m, nn.Conv2d) and m.kernel_size != (1,1):
        # TODO: 1x1 conv in resnet not supported
        mask_dict_conv[id(m)] = mask_list[conv_layer_id]
        conv_layer_id += 1
      elif isinstance(m, nn.BatchNorm2d) and len(mask_dict_bn) == len(mask_dict_conv) - 1:
        mask = mask_list[conv_layer_id-1].sum(dim=1)[:, 0, 0]
        mask = mask / mask.max()
        mask_dict_bn[id(m)] = mask
      # elif isinstance(m, nn.ReLU):
      #   mask = mask_list[conv_layer_id-1].sum(dim=1)[:, 0, 0]
      #   mask = mask / mask.max()
      #   mask_dict_relu[id(m)] = mask
      # elif isinstance(m, nn.MaxPool2d):
      #   mask = mask_list[conv_layer_id-1].sum(dim=1)[:, 0, 0]
      #   mask = mask / mask.max()
      #   mask_dict_pool[id(m)] = mask

    foo(model)
    input = Variable(torch.rand(3, 3, input_res, input_res), requires_grad = True).cuda()
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))

    print('  + Number of FLOPs for prnd model: %.8fG' % (total_flops / 3 / 1e9))
    # print('list_conv', list_conv)
    # print('list_linear', list_linear)
    # print('list_bn', list_bn)
    # print('list_relu', list_relu)
    # print('list_pooling', list_pooling)
    return total_flops

