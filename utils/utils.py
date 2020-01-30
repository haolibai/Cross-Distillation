""" This file stores some common functions for learners """

import torch
import torch.nn as nn
import pdb
import math
import numpy as np
import logging


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
  """ Code taken from torchvision """
  def __init__(self, num_batches, *meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def show(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    logging.info(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k
     Code taken from torchvision.
  """
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_seen_unseen(output, target, seen_classes, topk=(1.)):
  """ Compute the acc of seen classes and unseen classes seperately."""
  seen_classes = list(range(seen_classes))
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    seen_ind = torch.zeros(target.shape[0]).byte().cuda()
    unseen_ind = torch.ones(target.shape[0]).byte().cuda()
    for v in seen_classes:
      seen_ind += (target == v)
    unseen_ind -= seen_ind

    seen_correct = pred[:, seen_ind].eq(target[seen_ind].view(1, -1).expand_as(pred[:, seen_ind]))
    unseen_correct = pred[:, unseen_ind].eq(target[unseen_ind].view(1, -1).expand_as(pred[:, unseen_ind]))

    seen_num = seen_correct.shape[1]
    res = []
    seen_accs = []
    unseen_accs = []
    for k in topk:
        seen_correct_k = seen_correct[:k].view(-1).float().sum(0, keepdim=True)
        seen_accs.append(seen_correct_k.mul_(100.0 / (seen_num+1e-10)))
        unseen_correct_k = unseen_correct[:k].view(-1).float().sum(0, keepdim=True)
        unseen_accs.append(unseen_correct_k.mul_(100.0 / (batch_size-seen_num+1e-10)))
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res, seen_accs, unseen_accs, seen_num

def augment_mixup(x_fixed, y_fixed, folds=1):
  """ MixUp augmentation.
      According to results on vgg, mixup is far better than gaussian.
  """
  alpha = 1 # uniform distribution
  x_new = x_fixed
  y_new = y_fixed

  for i in range(folds):
    lam = np.random.beta(alpha, alpha) # all the data in the same batch share the same lam
    index = torch.randperm(x_fixed.shape[0])
    x_tmp = lam * x_fixed + (1. - lam) * x_fixed[index]
    y_tmp = lam * y_fixed + (1. - lam) * y_fixed[index]
    x_new = torch.cat([x_new, x_tmp], dim=0)
    y_new = torch.cat([y_new, y_tmp], dim=0)
  return x_new, y_new


def augment_gaussian(x_fixed, y_fixed, folds=1, std=1e-2):
  x_new = x_fixed
  y_new = y_fixed
  device = x_new.device
  for i in range(folds):
    noise = torch.randn(x_fixed.shape).to(device) * std
    x_tmp = x_fixed + noise
    y_tmp = y_fixed
    x_new = torch.cat([x_new, x_tmp], dim=0)
    y_new = torch.cat([y_new, y_tmp], dim=0)
  return x_new, y_new


def augment_repeat(loader, data_aug, folds=1):
  """ Repeat crop/rotate/flip opterations """
  assert data_aug, "Turn on data_aug! Loader does not produce change"
  xs, ys = [], []
  for i in range(folds):
    x, y = next(loader)
    xs.append(x)
    ys.append(y)
  x_new = torch.cat(xs, dim=0)
  y_new = torch.cat(ys, dim=0)
  return x_new, y_new


def calc_model_sparsity(model):
  ws = model.parameters()
  layerwise_spars = []
  num_elems = 0.
  num_nzero_elems = 0.
  for w in ws:
    if w.ndimension() == 4:
      if w.shape[2] == 1:
        # skip the downsample layers
        continue
      nonzero_elems = len(w.view(-1).nonzero())
      elems = w.nelement()
      layerwise_spars.append(float(nonzero_elems / elems))
      num_nzero_elems += nonzero_elems
      num_elems += elems

  overall_spars = num_nzero_elems / num_elems
  return layerwise_spars, overall_spars


def is_last_layer(layer):
  W = layer.weight
  if W.ndimension() == 2 and (W.shape[0] == 10 or W.shape[0] == 100 or W.shape[0] == 1000):
    return True
  else:
    return False


def is_first_layer(layer):
  if isinstance(layer, nn.Conv2d):
    W = layer.weight
    if W.ndimension() == 4 and (W.shape[1] == 3 or W.shape[1] == 1):
      return True
    else:
      return False
  else:
    return False


def reinitialize_conv_weights(model, init_first=False):
  """ Only re-initialize the kernels for conv layers """
  print("re-intializing conv weights done. evaluating...")
  for W in model.parameters():
    if W.ndimension() == 4:
      if W.shape[1] == 3 and not init_first:
        continue # do not init first conv layer
      nn.init.kaiming_uniform_(W, a = math.sqrt(5))


def weights_init(m):
  """ default param initializer in pytorch. """
  if isinstance(m, nn.Conv2d):
    n = m.in_channels
    for k in m.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    m.weight.data.uniform_(-stdv, stdv)
    if m.bias is not None:
        m.bias.data.uniform_(-stdv, stdv)
  elif isinstance(m, nn.Linear):
    stdv = 1. / math.sqrt(m.weight.size(1))
    m.weight.data.uniform_(-stdv, stdv)
    if m.bias is not None:
        m.bias.data.uniform_(-stdv, stdv)


def img2col(x, l):
  """ Test passed. Safe to use.
  Args:
    x: the feature map, in NCHW shape;
    l: the conv layer;
  Return:
    x_mat: matrized feature map, shape of [N * H * W, c * kh * kw];
  """
  assert isinstance(l, nn.Conv2d) and x.ndimension() == 4, 'the input feature map must be in shape [N, C, H, W]'
  # pad the feature map
  W = next(l.parameters())
  kh, kw = W.shape[2], W.shape[3]
  ph, pw = l.padding # kernel size
  sh, sw = l.stride # stride
  if ph == 0 and pw == 0:
    x_pad = x
  else:
    x_pad = nn.functional.pad(x, pad=(ph, ph, pw, pw))

  n, c, h, w = x_pad.shape
  nb_iter_col = (h - kh + 1) // sh
  nb_iter_row = (w - kw + 1) // sw

  x_list = [x_pad[:, i, :, :] for i in range(c)]
  x_vec_list = []
  for x_c in x_list:
    # x_c: NHW
    tmp_list = []
    for i in range(nb_iter_col):
      i = i * sh
      for j in range(nb_iter_row):
        j = j * sw
        patches = x_c[:, i:i+kh, j:j+kw].contiguous().view(n, 1, kh*kw)
        tmp_list.append(patches)
    x_c = torch.cat(tmp_list, dim=1)  # [N, itr_h x itr_w, kh x kw]
    x_c = x_c.view(nb_iter_col * nb_iter_row * n, kh * kw)  # [N x itr_h x itr_w, kh x kw]
    assert x_c.shape[0] == n*nb_iter_col*nb_iter_row and x_c.shape[1] == kh*kw, 'shape mismatch'
    x_vec_list.append(x_c)

  # reshape the list to a matrix
  x_list = [x.unsqueeze(dim=1) for x in x_vec_list] # [N x itr_h x itr_w, 1, kh x kw]
  x_mat = torch.cat(x_list, dim=1).view(-1, c * W.shape[2] * W.shape[3])
  return x_mat


def unit_test_img2col():
  """ Test passed. Safe to use. """
  x = torch.rand(128, 32, 8, 8).cuda()
  # you may tri different conv layers for test
  l = nn.Conv2d(32, 64, kernel_size=5, padding=3, stride=2, bias=False).cuda()
  n, c, _, _, = x.shape
  W = next(l.parameters())
  x_mat = img2col(x, l)
  y = l(x)
  y_mat = x_mat.mm(W.view(2*c, -1).t()) # [N x H x W, co]
  y_recon = y_mat.view(n, y.shape[2], y.shape[3], 2*c).permute(0, 3, 1, 2)  # [N x co x H x W]
  norm = torch.norm(y_recon - y, p=2)
  assert norm < 1e-2, 'failed to recover from img2col'
  print(norm.item())
  return norm

if __name__ == '__main__':
  unit_test_img2col()
  print('done')
