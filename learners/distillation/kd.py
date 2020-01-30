import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


def get_distillation_loss(student, teacher, X, temp, kd_type='ce'):
  assert kd_type in ['ce', 'l2'], 'unknown distillation type'
  stu_logits = student(X)
  tea_logits = teacher(X).detach() # soft labels
  if kd_type == 'ce':
    kd_loss = cross_entropy_loss(stu_logits, tea_logits, temp)
  elif kd_type == 'l2':
    kd_loss = torch.sum((stu_logits-tea_logits)**2, dim=1).mean()
  return kd_loss


def cross_entropy_loss(stu_logits, tea_logits, temp=1.):
  """ the same as nn.CrossEntropyLoss, but more flexible
  Args:
    stu_logits: tensor of shape [N, class]
    tea_logits: tensor of shape [N, class]
    temp: the distillation temperature
  Return:
    kd_loss: the cross entropy on soft labels
  """
  pred = torch.nn.functional.softmax(stu_logits, dim=1)
  labels = torch.nn.functional.softmax(tea_logits / temp, dim=1)
  kd_loss = (-labels * torch.log(pred)).sum(dim=1).mean()
  return kd_loss


def criterion_alternative_L2(source, target, margin=1.):
    loss = ((source + margin)**2 * ((source > -margin) & (target <= 0)).float() +
            (source - margin)**2 * ((source <= margin) & (target > 0)).float())
    return torch.abs(loss).sum()


def criterion_L2(source, target, selective=False):
    """ Similar to FitNet """
    loss = ((source - target) ** 2)
    if selective:
      # only penalize student and teacher feature maps larger than 0s
      mask = (source > 0).float() * (target > 0).float()
      loss *= mask
    return loss.sum()


def criterion_attention_L2(source, target):
    """ Similar to Attention transfer """
    source_post = F.normalize(source.pow(2).mean(1).view(source.size(0), -1))
    target_post = F.normalize(target.pow(2).mean(1).view(target.size(0), -1))
    loss = (source_post - target_post).pow(2).sum()
    return loss


def criterion_hinge_loss(source, target, sqaure=False):
    """ Apply min(0, -source \dot target)
    # NOTE: it seems sqaure it is better, but the similarity is not that
    """
    vec = source * target
    mask = (vec < 0).detach().float()
    loss = (- mask * vec)
    if sqaure:
      loss = loss.pow(2)
    return loss.sum()


def similarity(student, teacher):
    similarity = 1. - ((student > 0) ^ (teacher > 0)).sum().float().item() / student.nelement()
    return similarity


def test_cross_entropy():
  logits = torch.rand(128, 10)
  labels = torch.randint(low=0, high=9, size=(128,))
  ground_loss = nn.CrossEntropyLoss()(logits, labels)

  ind = [torch.arange(128), labels]
  soft_labels = torch.zeros(128, 10)
  soft_labels[ind] = 1.
  kd_loss = cross_entropy_loss(logits, soft_labels)

if __name__ == '__main__':
  test_cross_entropy()
  print('done')
