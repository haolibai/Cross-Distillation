import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
from learners.basic_learner import BasicLearner
import math
import numpy as np
import time
import copy
from learners.distillation.kd import *
from utils.utils import *
import pdb


class WeightPruneLearner(BasicLearner):
  """ Perform weight pruning (non-structural) and fine-tuning """

  def __init__(self, model, loaders, args, device):
    super(WeightPruneLearner, self).__init__(model, loaders, args, device)
    self.teacher_model = copy.deepcopy(self.model)
    self.teacher_model.load_state_dict(torch.load(self.args.load_path))
    self.teacher_model.train()
    self.setup_optim()  # over-ride the original one
    self.save_path_pruned = os.path.join(os.path.dirname(self.args.load_path), str(self.args.target_ratio) + '_unstructure_pruned.pt')

  def evaluate(self, is_train=False, epoch=None):
    self.switch_mode('eval')
    if not is_train:
      self.model.load_state_dict(torch.load(self.load_path))
      logging.info("Model successfully restored from %s" % self.load_path)
    if self.args.no_classwise and self.args.classwise_seen_unseen:
      test_acc, test_loss = self.epoch_seen_unseen()
    else:
      test_acc, test_loss = self.epoch(False)
    return test_acc, test_loss

  def finetune(self):
    # self.evaluate(is_train=False)
    self.model.load_state_dict(torch.load(self.load_path))
    logging.info("Model successfully restored from %s" % self.load_path)
    # self.evaluate(True, None)

    logging.info("Job Id: %s, Start Pruning..., Target Ratio: %s" % (self.args.job_id, self.args.target_ratio))

    self.__progressive_layerwise_prune()
    test_acc, _ = self.evaluate(is_train=True)

    layerwise_spars, overall_spars = calc_model_sparsity(self.model)
    print("Unstructured pruning done. layerwise spars:", layerwise_spars)
    print("Overall sparsity: %.4f", overall_spars)
    torch.save(self.model.state_dict(), self.save_path_pruned)
    logging.info("Prund models saved at %s, start fine-tuning..." % self.save_path_pruned)

    # start fine-tuning
    for epoch in range(self.args.epochs):
      self.switch_mode('train')
      logging.info("Finetuning at Epoch %d" % epoch)
      ft_err, ft_loss = self.epoch(True)
      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)
        # save the model
        torch.save(self.model.state_dict(), self.save_path_pruned)
        logging.info("Model stored at: " + self.save_path_pruned)

  def epoch(self, is_train):
    """ Rewrite this function if necessary in the sub-classes. """

    loader = self.train_loader if is_train else self.test_loader

    # setup statistics
    batch_time = AverageMeter('Time', ':3.3f')
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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

      batch_time.update(time.time() - end)
      end = time.time()

      # show the training/evaluating statistics
      if (idx % self.args.print_freq == 0) or (idx+1) % loader_len == 0:
        progress.show(idx)

      if self.args.use_few_data and is_train and (idx+1) == loader_len:
        # Stop infinite loop
        break

    return top1.avg, losses.avg

  def epoch_seen_unseen(self):
    """ only used for evaluation with classwise=False and classewise_seen_unseen=True"""
    loader = self.test_loader

    # setup statistics
    batch_time = AverageMeter('Time', ':3.3f')
    losses = AverageMeter('Loss', ':.4e')
    all_top1 = AverageMeter('Acc@1', ':3.3f')
    all_top5 = AverageMeter('Acc@5', ':3.3f')
    seen_top1 = AverageMeter('Acc_S@1', ':3.3f')
    seen_top5 = AverageMeter('Acc_S@5', ':3.3f')
    unseen_top1 = AverageMeter('Acc_US@1', ':3.3f')
    unseen_top5 = AverageMeter('Acc_US@5', ':3.3f')
    metrics = [batch_time, all_top1, all_top5, seen_top1, seen_top5, unseen_top1, unseen_top5, losses]

    loader_len = len(loader)
    progress = ProgressMeter(loader_len, *metrics, prefix='Job id: %s, ' % self.args.job_id)
    end = time.time()

    for idx, (X, y) in enumerate(loader):

      # data_time.update(time.time() - end)
      X, y = X.to(self.device), y.to(self.device)
      yp = self.model(X)
      loss = nn.CrossEntropyLoss()(yp, y)

      all_accs, seen_accs, unseen_accs, seen_num = accuracy_seen_unseen(yp, y, self.args.num_data, topk=(1, 5))

      all_top1.update(all_accs[0].item(), X.shape[0])
      all_top5.update(all_accs[1].item(), X.shape[0])
      seen_top1.update(seen_accs[0].item(), seen_num+1e-10)
      seen_top5.update(seen_accs[1].item(), seen_num+1e-10)
      unseen_top1.update(unseen_accs[0].item(), X.shape[0]-seen_num+1e-10)
      unseen_top5.update(unseen_accs[1].item(), X.shape[0]-seen_num+1e-10)

      losses.update(loss.item(), X.shape[0])
      batch_time.update(time.time() - end)
      end = time.time()

      # show the training/evaluating statistics
      if (idx % self.args.print_freq == 0) or (idx+1) % loader_len == 0:
        progress.show(idx)

    return all_top1.avg, losses.avg

  def __progressive_layerwise_prune(self):
    logging.info("Layerwise progressive pruning...")
    conv_layer_id = 0
    self.mask_list = []
    for idx, (l_prnd, l_full) in enumerate(zip(self.model.modules(), self.teacher_model.modules())):
      if self.__check_prunable(l_prnd, l_full):
        if is_first_layer(l_prnd) and not self.args.prune_first_layer:
          weight = l_prnd.weight
          mask = torch.ones_like(weight).to(self.device)
        else:
          print("Layerwise pruning for layer...", l_prnd)
          mask = self.__solve_lst_pgd(l_prnd, l_full)
        conv_layer_id += 1
        self.mask_list.append(mask)
        if conv_layer_id % 10 == 0:
          test_acc, _ = self.evaluate(True, None)

  def __solve_lst_pgd(self, l_prnd, l_full, verbose=True):
    """ Solve the least sqaure problem with proximal gradient descent,
        and prune the coresponding weights to 0s.

    Inputs:
      l_prnd: the conv layer of student model
      l_full: the conv layer of teacher model
      verbose: True if print info.
    Return:
      mask: a tensor, shape: [co * cin * k * k], the same as weights
    """
    def proximal_map(W, ratio):
      vec_W = W.clone().view(-1)
      truncate_idx = int(vec_W.shape[0] * ratio)
      thresh = torch.topk(vec_W.abs(), k=truncate_idx)[0][-1]
      mask = (W.abs() > thresh).float()
      W.data *= mask
      return mask

    def mask_grad(opt, weight, mask):
      weight.grad.data *= mask
      if len(opt.state[weight].keys()) > 0:
        opt.state[weight]['exp_avg'] *= mask
        opt.state[weight]['exp_avg_sq'] *= mask

    Xs, Ys, Xt, Yt = self.__get_4D_input_output(l_prnd, l_full)
    l_prnd_copy = copy.deepcopy(l_prnd)
    l_full_copy = copy.deepcopy(l_full)
    weight = l_prnd_copy.weight
    n, co, h, w = Yt.shape
    _, ci, k, k = weight.shape

    # initial configurations
    lr = self.args.pgd_lr
    nb_iters = self.args.pgd_iters
    opt = optim.Adam([weight], lr=lr)

    mask = torch.ones_like(weight).to(self.device)
    for i in range(nb_iters):

      if self.args.num_data > self.args.batch_size or not self.args.use_few_data:
        # cannot load the data in one batch
        Xs, Ys, Xt, Yt = self.__get_4D_input_output(l_prnd, l_full)

      # gradient descent step
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

      opt.zero_grad()
      loss.backward()
      mask_grad(opt, weight, mask)
      opt.step()

      # proximal step
      if i <= (nb_iters // 3) and not self.args.pgd_once:
        ratio = self.args.target_ratio + (1 - self.args.target_ratio) * ( (nb_iters//3 - i) / (nb_iters//3))
        mask = proximal_map(weight, ratio)

      if (i+1) % (nb_iters//5) == 0 and verbose:
        print("Iter:%d, ratio: %.3f, lr:%.6f, loss:%.8f" % (i+1, ratio, lr, loss.item()))

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
      # x, _ = augment_gaussian(x, _, folds=10)
      # x, _ = augment_repeat(loader, self.args.data_aug, folds=self.args.augment_folds)

    self.model(x)
    self.teacher_model(x)

    x_prnd, y_prnd = pair_prnd[0]
    x_full, y_full = pair_full[0]

    # remove the hooks
    hook_prnd.remove()
    hook_full.remove()
    return x_prnd, y_prnd, x_full, y_full

  def __mask_grad(self):
    mask_idx = 0
    for m in self.model.modules():
      if isinstance(m, nn.Conv2d):
        m.weight.grad.data *= self.mask_list[mask_idx]
        mask_idx += 1

  def __check_prunable(self, l_prnd, l_full):
    if isinstance(l_prnd, nn.Conv2d) and isinstance(l_full, nn.Conv2d):
      return True
    else:
      return False
