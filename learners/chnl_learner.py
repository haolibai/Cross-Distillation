import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learners.basic_learner import BasicLearner
from utils.utils import *
from learners.distillation.kd import *
import math
import pdb
import copy
import numpy as np
import inspect
import time
import torch.nn.functional as F

""" Data dependent channel pruning.
    Cross distillation is incorporated.
"""


class ChannelLearner(BasicLearner):
  def __init__(self, model, loaders, args, device):
    super(ChannelLearner, self).__init__(model, loaders, args, device)
    self.teacher_model = copy.deepcopy(self.model)
    self.teacher_model.load_state_dict(torch.load(self.args.load_path))
    self.setup_optim()  # over-ride the original one
    self.save_path_pruned = os.path.join(os.path.dirname(self.args.load_path), str(self.args.target_ratio) + '_chnl_pruned.pt')

  def train(self):
    # NOTE: prune from scratch
    self.__layerwise_prune()
    for epoch in range(self.args.epochs):
      self.switch_mode('train')
      logging.info("Training at Epoch: %d" % epoch)
      train_acc, train_entropy = self.epoch(True)

      if self.lr_scheduler:
        self.lr_scheduler.step()

      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)

        torch.save(self.model.state_dict(), self.save_path)
        logging.info("Model stored at: " + self.save_path)

  def evaluate(self, is_train=False, epoch=None):
    self.switch_mode('eval')
    if not is_train:
      self.model.load_state_dict(torch.load(self.load_path))

    test_acc, test_entropy = self.epoch(False)
    return test_acc, test_entropy

  def finetune(self):
    self.model.load_state_dict(torch.load(self.load_path))
    self.evaluate(True, None)

    self.params = [w for w in self.model.parameters()]
    self.__layerwise_prune()

    # calc sparsity
    self.mask_list = self.__remove_rest_params()
    layerwise_spars, overall_spars = calc_model_sparsity(self.model)
    print('layerwise sparsity', layerwise_spars)
    print('overall sparsity: %.4f' % overall_spars)
    torch.save(self.model.state_dict(), self.save_path_pruned)
    logging.info("Just pruned model saved at %s" % self.save_path_pruned)

    test_acc, _ = self.evaluate(True, None)

    logging.info("Model restored from %s, start channel pruning with finetuning..." % (self.load_path))
    for epoch in range(self.args.epochs):
      self.switch_mode('train')

      logging.info("Finetune at Epoch: %d" % epoch)
      ft_err, ft_entropy = self.epoch(True)

      # NOTE: stop lr decaying
      if self.lr_scheduler:
        self.lr_scheduler.step()

      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        logging.info("Evaluation at Epoch: %d" % epoch)
        test_acc, _ = self.evaluate(True, epoch)

        torch.save(self.model.state_dict(), self.save_path)
        logging.info("Model stored at: " + self.save_path)

    layerwise_spars, overall_spars = calc_model_sparsity(self.model)
    print('layerwise sparsity', layerwise_spars)
    print('overall sparsity: %.4f' % overall_spars)

  def epoch(self, is_train):
    """ Rewrite this function if necessary in the sub-classes. """

    loader = self.train_loader if is_train else self.test_loader

    # setup statistics
    batch_time = AverageMeter('Time', ':3.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':3.3f')
    top5 = AverageMeter('Acc@5', ':3.3f')
    metrics = [batch_time, top1, top5, losses]

    if self.args.use_kd:
      kd_losses = AverageMeter('KD Loss:', ':4e')
      metrics.append(kd_losses)

    loader_len = int(self.args.num_data / self.args.batch_size)+1 if is_train and self.args.use_few_data  else len(loader)

    progress = ProgressMeter(loader_len, *metrics, prefix='Job id: %s, ' % self.args.job_id)
    end = time.time()

    for idx, (X, y) in enumerate(loader):

      # data_time.update(time.time() - end)
      X, y = X.to(self.device), y.to(self.device)
      yp = self.model(X)
      loss = nn.CrossEntropyLoss()(yp, y)

      if is_train and self.args.use_kd:
        kd_loss = self.args.kd_regu * get_distillation_loss(self.model, self.teacher_model, X, self.args.kd_temp, self.args.kd_type)
        loss += kd_loss
        kd_losses.update(kd_loss.item(), X.shape[0])

      acc1, acc5 = accuracy(yp, y, topk=(1, 5))
      top1.update(acc1[0], X.shape[0])
      top5.update(acc5[0], X.shape[0])
      losses.update(loss.item(), X.shape[0])

      if is_train:
        self.__optimize(loss)

      batch_time.update(time.time() - end)
      end = time.time()

      # show the training/evaluating statistics
      if (idx % self.args.print_freq == 0) or (idx+1) % (loader_len) == 0:
        progress.show(idx)

      if self.args.use_few_data and is_train and (idx+1) == loader_len:
        # Stop infinite loop
        break

    return top1.avg, losses.avg

  def __optimize(self, loss):
    """ A single updt step """
    self.opt.zero_grad()
    loss.backward()
    self.__mask_grad()
    self.opt.step()

  def __mask_grad(self):
    # TODO: mask the BN params
    mask_idx = 0
    for m in self.model.modules():
      if isinstance(m, nn.Conv2d):
        if m.kernel_size == (1, 1):
          continue

        m.weight.grad.data *= self.mask_list[mask_idx]
        mask_idx += 1

  def __layerwise_val_loss(self, l_prnd, l_full):
    nb_epochs = len(self.test_loader) - 1
    loss_list = []
    self.iter_test_loader = iter(self.test_loader)
    for i in range(nb_epochs):
      Xs, Ys, Xt, Yt = self.__get_4D_input_output(l_prnd, l_full, is_train=False)
      n, co, h, w = Ys.shape
      loss  = criterion_L2(Ys, Yt)/(n*co*h*w)
      loss_list.append(loss.item())
    print("The averaged validation loss: %.8f" % (sum(loss_list)/nb_epochs))

  def __remove_rest_params(self, verbose=True):
    """ After in-channel pruning, we can safely remove the redundant out channels and bn params that are useless for the next layer.
        This operation should not influence the performance.
        The evaluation acc is the same before and after this function. Check passed.
    """
    conv_layer_id = 0
    final_mask_list = []
    for m in self.model.modules():
      if isinstance(m, nn.Conv2d):

        if m.kernel_size == (1, 1):
          # skip the downsample layers
          continue

        W = m.weight
        if conv_layer_id  < len(self.mask_list) - 1:
          mask_in = torch.zeros_like(W).to(self.device)
          mask_out = torch.zeros_like(W).to(self.device)
          nzero_in = self.mask_list[conv_layer_id].sum(dim=0)[:, 0, 0].nonzero().squeeze()
          nzero_out = self.mask_list[conv_layer_id+1].sum(dim=0)[:, 0, 0].nonzero().squeeze()
          mask_in[:, nzero_in, :] += 1
          mask_out[nzero_out, :] += 1
          mask = mask_in * mask_out

          W.data *= mask
          final_mask_list.append(mask)
          conv_layer_id += 1

          if verbose:
            print("removing out_channels for kernels:", W.shape)
        else:
          assert conv_layer_id == len(self.mask_list) - 1
          final_mask_list.append(self.mask_list[conv_layer_id])

      elif isinstance(m, nn.BatchNorm2d) and conv_layer_id < len(self.mask_list) - 1:
        # NOTE: in the last conv layer, do not turn off the BN params since
        # cout is not changed.
        gamma, beta = m.weight, m.bias
        mean, std = m.running_mean, m.running_var
        mask = self.mask_list[conv_layer_id].sum(dim=0)[:, 0, 0]
        mask = mask / mask.max()
        gamma.data *= mask
        beta.data *= mask
        mean.data *= mask
        std.data *= mask

    print("Remove rest out channels done")
    return final_mask_list

  def __layerwise_prune(self):
    """ perform layerwise regression + cross distillation. After pruning, the 'removed channels' are set to 0s
    Return:
      mask_list: the mask of channels for pruned layers. For gradient masking in finetuning.
    """

    self.mask_list = []
    self.cfgs = self.__get_cfgs()
    conv_layer_id = 0

    for idx, (l_prnd, l_full) in enumerate(zip(self.model.modules(), self.teacher_model.modules())):

      if self.__check_prunable(l_prnd, l_full):

        # TODO: at least for resnet, eval is better than train.
        if self.args.model_type.startswith('resnet'):
          self.model.eval()
          self.teacher_model.eval()
        else:
          self.model.train()
          self.teacher_model.train()

        conv_layer_id = conv_layer_id + 1 if self.cfgs[conv_layer_id] == 'M' else conv_layer_id
        assert l_prnd.in_channels >= self.cfgs[conv_layer_id], "in channels %d smaller than %d" % (l_prnd.in_channels, self.cfgs[conv_layer_id])

        if is_first_layer(l_prnd) and not self.args.prune_first_layer:
          # skip the first layer
          mask = torch.ones_like(l_prnd.weight)
        else:
          print("channel pruning for layer ", l_prnd)
          mask = self.__solve_lst_pgd(l_prnd, l_full, conv_layer_id)
          # self.__layerwise_val_loss(l_prnd, l_full)

        self.mask_list.append(mask)
        print("channel pruning done for layer:", l_prnd)
        conv_layer_id += 1

        # if (conv_layer_id+1) % 8 == 0:
        #   self.evaluate(True, None)

    # assert conv_layer_id == len(self.cfgs), 'some cfg are not used!'

  def __solve_lst_pgd(self, l_prnd, l_full, conv_layer_id, verbose=True):
    """ Solve the least sqaure problem with proximal gradient descent,
        and prune the coresponding weights to 0s.

    Inputs:
      X, Y: tensors of input and output, in 4-D shape;
      l_prnd: the conv layer of student model
      verbose: True if print info.
    Return:
      mask: a tensor, shape: [cin * k * k], the mask on cols.
    """
    def proximal_map(W, num_left):
      norms = torch.norm(W.permute(1,0,2,3).contiguous().view(ci, -1), p=2, dim=1)
      thresh = torch.topk(norms, k=num_left)[0][-1] #(values, indices)
      for i in range(ci):
        norm = torch.norm(W.data[:, i, :]/thresh, p=2)
        W.data[:, i, :] = W.data[:, i, :] if norm.item() >= 1 else torch.zeros(co,k,k).to(self.device)

    def mask_grad(opt, weight, mask):
      weight.grad.data *= mask
      if len(opt.state[weight].keys()) > 0:
        opt.state[weight]['exp_avg'] *= mask
        opt.state[weight]['exp_avg_sq'] *= mask

    cfg = self.cfgs[conv_layer_id]
    Xs, Ys, Xt, Yt = self.__get_4D_input_output(l_prnd, l_full)
    l_prnd_copy = copy.deepcopy(l_prnd)
    l_full_copy = copy.deepcopy(l_full)
    weight = l_prnd_copy.weight
    n, co, h, w = Yt.shape
    _, ci, k, k = weight.shape

    # initial configurations
    # for VGG-16 cifar: 5e-4 1000 iters; For Resnet-56, 1e-5, 2000 iters
    lr = self.args.pgd_lr
    nb_iters = self.args.pgd_iters

    opt = optim.Adam([weight], lr=lr)
    channels_left_prev = ci
    mask = torch.ones_like(weight).to(self.device)

    for i in range(nb_iters):

      if self.args.num_data > self.args.batch_size or not self.args.use_few_data:
        # cannot load the data in one batch
        Xs, _, Xt, _ = self.__get_4D_input_output(l_prnd, l_full)

      Yst = l_full_copy(Xs)
      Yts = l_prnd_copy(Xt)
      Ytt = l_full_copy(Xt)
      Yss = l_prnd_copy(Xs)

      if self.args.use_cvx:
        Lc = criterion_L2(Yts, Ytt) / (n*co*h*w)
        Li = criterion_L2(Yst, Yss) / (n*co*h*w)
        loss = self.args.mu * Lc + (1. - self.args.mu) * Li
      else:
        loss = criterion_L2(Yss, Ytt) / (n*co*h*w)

      # gradient descent step
      opt.zero_grad()
      loss.backward()
      mask_grad(opt, weight, mask)
      opt.step()

      # proximal step
      if i <= (nb_iters // 3) and not self.args.pgd_once:
        channels_left = int((ci - cfg) * float((nb_iters/3 - i) / (nb_iters/3))) + cfg
        if channels_left < channels_left_prev:
          proximal_map(weight, channels_left)
          channels_left_prev = channels_left
        nzero_idx = weight.sum(dim=0)[:, 0, 0].nonzero().squeeze()
        mask.zero_()
        mask[:, nzero_idx, :, :] += 1

      elif i == 0 and self.args.pgd_once:
        channels_left = cfg
        proximal_map(weight, channels_left)
        nzero_idx = weight.sum(dim=0)[:, 0, 0].nonzero().squeeze()
        mask = torch.zeros_like(weight).to(self.device)
        mask[:, nzero_idx, :, :] += 1

      else:
        for param in opt.param_groups:
          param['lr'] = lr * 1e-1

      if (i) % (nb_iters//5 - 1) == 0 and verbose:
        print("Iter:%d, channels_left: %d, lr:%.6f, loss:%.8f" % \
            (i+1, channels_left, lr, loss.item()))

    # assign the value back
    l_prnd.weight.data = weight
    return mask

  def __get_4D_input_output(self, l_prnd, l_full, is_train=True):
    """ For pgd solver
    Return:
        x_prnd: a 4-D tensor of the (l-1)'s student feature map
        y_full: a 4-D tensor of l's teacehr feature map
    """
    # NOTE: remember to detach the variables from the autograd graph, only
    # solve the current layer
    def get_hidden_prnd(module, input_, output_):
      pair_prnd.append((input_[0].detach(), output_.detach()))

    def get_hidden_full(module, input_, output_):
      pair_full.append((input_[0].detach(), output_.detach()))

    loader = self.train_loader if is_train else self.iter_test_loader
    # add the hooks
    hook_prnd = l_prnd.register_forward_hook(get_hidden_prnd)
    hook_full = l_full.register_forward_hook(get_hidden_full)

    pair_prnd = []
    pair_full = []

    assert self.args.use_few_data
    x, t = next(loader)
    if self.args.further_augment:
      x, t = augment_mixup(x, t, folds=self.args.augment_folds)
      # x, _ = augment_gaussian(x, t, folds=10)
      # x, _ = augment_repeat(loader, self.args.data_aug, folds=self.args.augment_folds)

    self.model(x)
    self.teacher_model(x)

    x_prnd, y_prnd = pair_prnd[0]
    x_full, y_full = pair_full[0]

    # remove the hooks
    hook_prnd.remove()
    hook_full.remove()
    return x_prnd, y_prnd, x_full, y_full

  def __check_prunable(self, l_prnd, l_full):
    if isinstance(l_prnd, nn.Conv2d) and isinstance(l_full, nn.Conv2d):
      if l_prnd.kernel_size == (1, 1):
        return False
      else:
        return True
    else:
      return False

  def __get_cfgs(self):
    """ Note that the cfgs are the number of in_channels in each conv layer """
    if self.args.model_type.startswith('vgg_16') and self.args.dataset == 'cifar10':
      # default = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512] out channels
      cfg_0 = [3, 64, 'M', 64, 128, 'M', 128, 256, 256, 'M', 256, 512, 512, 'M', 512, 512, 512]
      cfg_same = [int(c * self.args.target_ratio) if c != 'M' else c for c in cfg_0]
      cfg_A = [3, 32, 'M', 64, 128, 'M', 128, 256, 256, 'M', 256, 256, 256, 'M', 256, 256, 256]
      cfg_B = [3, 26, 'M', 52, 103, 'M', 103, 205, 205, 'M', 205, 205, 256, 'M', 205, 205, 205]
      cfg_C = [3, 26, 'M', 32, 64, 'M', 64, 128, 128, 'M', 128, 128, 128, 'M', 205, 205, 205]
      return cfg_same

    elif self.args.model_type.startswith('vgg_19') and self.args.dataset == 'ilsvrc_12':
      # defaul [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
      cfg_0 = [3, 64, 'M', 64, 128, 'M', 128, 256, 256, 256, 'M', 256, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
      cfg_same = [int(c * self.args.target_ratio) if c != 'M' else c for c in cfg_0]
      return cfg_same

    elif self.args.model_type.startswith('vgg_16') and self.args.dataset == 'ilsvrc_12':
      cfg_0 = [3, 64, 'M', 64, 128, 'M', 128, 256, 256, 'M', 256, 512, 512, 'M', 512, 512, 512]
      cfg_same = [int(c * self.args.target_ratio) if c != 'M' else c for c in cfg_0]
      cfg_2x = [3, 35, 'M', 35, 70, 'M', 70, 140, 140, 'M', 140, 422, 422, 'M', 422, 512, 512]
      cfg_4x = [3, 24, 'M', 26, 41, 'M', 58, 108, 108, 'M', 128, 184, 276, 'M', 276, 512, 512]
      cfg_5x = [3, 24, 'M', 22, 41, 'M', 51, 108, 89, 'M', 111, 184, 276, 'M', 228, 512, 512]
      return cfg_4x

    elif self.args.model_type.startswith('resnet_56') and self.args.dataset == 'cifar10':
      skip = [0, 16, 20, 38, 54]
      # target_ratios = [0.25, 0.5, 0.7]
      target_ratios = [self.args.target_ratio] * 3
      layer_id = 0
      cfg_skip = []
      for m in self.model.modules():
        if isinstance(m, nn.Conv2d):
          in_channels = m.in_channels
          if layer_id % 2 == 0 and layer_id not in skip:
            if layer_id <= 18:
                stage = 0
            elif layer_id <= 36:
                stage = 1
            else:
                stage = 2
            cfg_skip.append(int(target_ratios[stage]*in_channels)+1)
            layer_id += 1
            continue
          else:
            cfg_skip.append(in_channels)
            layer_id += 1
      return cfg_skip

    elif self.args.model_type.startswith('resnet_34') and self.args.dataset == 'ilsvrc_12':
      skip = [0, 6, 12, 14, 24]
      target_ratios = [self.args.target_ratio]*3 + [1.]
      layer_id = 0
      cfg_skip = []
      for m in self.model.modules():
        if isinstance(m, nn.Conv2d):

          if m.kernel_size == (1,1):
            continue

          in_channels = m.in_channels
          if layer_id % 2 == 0 and layer_id not in skip:
            if layer_id <= 6:
              stage = 0
            elif layer_id <= 12:
              stage = 1
            elif layer_id <= 24:
              stage = 2
            else:
              stage = 3
            num_channels_left = int(target_ratios[stage]*in_channels)
            cfg_skip.append(num_channels_left)
            print("%d channels left for module" % num_channels_left, m)
            layer_id += 1
          else:
            cfg_skip.append(in_channels)
            layer_id += 1
      return cfg_skip
    else:
      raise NotImplementedError

