import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import pdb
from learners.distillation.kd import *
from utils.utils import *
from timeit import default_timer as timer
import time
import copy
from utils.compute_flops import *


class BasicLearner(object):
  """ Performs vanilla training """
  def __init__(self, model, loaders, args, device):
    self.args = args
    self.device = device
    self.model = model
    self.__build_path()
    self.train_loader, self.test_loader = loaders
    self.setup_optim()
    self.num_classes = self.__get_num_classes()
    if self.args.use_kd:
      assert self.args.kd_load_path, 'You must assign the load path for the teacher model'
      with torch.no_grad():
        # TODO: add different types of teacher model
        self.teacher_model = copy.deepcopy(model)
        self.teacher_model.load_state_dict(torch.load(self.args.kd_load_path))
        logging.info("Teacher model restored from %s" % self.args.kd_load_path)

  def train(self):
    self.warm_up_lr()

    for epoch in range(self.args.epochs):
      self.switch_mode('train')
      logging.info("Training at Epoch: %d" % epoch)
      train_acc, train_loss = self.epoch(True)

      if self.lr_scheduler:
        self.lr_scheduler.step()

      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)

        # save the model
        torch.save(self.model.state_dict(), self.save_path)
        logging.info("Model stored at: " + self.save_path)

  def evaluate(self, is_train=False, epoch=None):
    self.switch_mode('eval')
    if not is_train:
      self.model.load_state_dict(torch.load(self.load_path))
      logging.info("Model successfully restored from %s" % self.load_path)
    test_acc, test_loss = self.epoch(False)
    return test_acc, test_loss

  def finetune(self):
    self.model.load_state_dict(torch.load(self.load_path))
    self.evaluate()
    logging.info("Model restored from %s, start finetuning" % (self.load_path))
    for epoch in range(self.args.epochs):
      self.switch_mode('trian')
      ft_acc, ft_loss = self.epoch(True)

      # NOTE: use the preset learning rate for all epochs.
      # if self.lr_scheduler:
      #   self.lr_scheduler.step()

      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)

        # save the model
        torch.save(self.model.state_dict(), self.save_path)
        logging.info("Model stored at: " + self.save_path)

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
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

      batch_time.update(time.time() - end)
      end = time.time()

      # show the training/evaluating statistics
      if (idx % self.args.print_freq == 0) or (idx+1) % loader_len == 0:
        progress.show(idx)

        if self.args.use_few_data and is_train:
          # Stop infinite loop
          break

    return top1.avg, losses.avg

  def switch_mode(self, mode):
    if mode == 'train':
      self.model.train()
    elif mode == 'eval':
      self.model.eval()
    else:
      raise ValueError("Unknown mode")

  def setup_optim(self):
    if self.args.model_type.startswith('model_'):
      self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr, \
          momentum=self.args.momentum, nesterov=self.args.nesterov, \
          weight_decay=self.args.weight_decay)
      # self.lr_scheduler = None
      self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[int(self.args.epochs*0.5), int(self.args.epochs*0.75)])
    elif self.args.model_type.startswith('resnet_') or self.args.model_type.startswith('vgg_'):
      self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr, \
          momentum=self.args.momentum, nesterov=self.args.nesterov, \
          weight_decay=self.args.weight_decay)
      self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[int(self.args.epochs*0.5), int(self.args.epochs*0.75)])
    else:
      raise ValueError("Unknown model, failed to initalize optim")

  def calc_param_size_flops(self):
    print_model_param_nums(self.model, multiply_adds=False)
    input_res = 32 if self.args.dataset.startswith('cifar') else 224
    print_model_param_flops(self.model, input_res, multiply_adds=False)
    try:
      print_pruned_param_nums(self.model, self.mask_list, multiply_adds=False)
      print_pruned_param_flops(self.model, input_res, self.mask_list, multiply_adds=False)
    except AttributeError:
      pass
    logging.info("Calc model param flops done.")

  def __build_path(self):
    extra = 'few' if self.args.use_few_data else 'full'
    if self.args.exec_mode == 'finetune':
      self.load_path = self.args.load_path
      self.save_path = os.path.join(os.path.dirname(self.load_path), 'model_ft.pt')
    elif self.args.exec_mode == 'train':
      self.save_path = os.path.join(self.args.save_path, '_'.join([self.args.model_type, self.args.learner, extra]), self.args.job_id, 'model.pt')
      self.load_path = self.save_path
    else:
      self.load_path = self.args.load_path
      self.save_path = self.load_path

  def warm_up_lr(self):
    if self.args.model_type.endswith('1202') or self.args.model_type.endswith('110'):
      for param in self.opt.param_groups:
        param['lr'] = 0.1 * self.args.lr
    else:
      pass

  def __get_num_classes(self):
    if self.args.dataset in ['cifar10']:
      return 10
    elif self.args.dataset == 'ilsvrc_12':
      return 1000


