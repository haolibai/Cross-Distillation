from nets.resnet_18s import *
from nets.resnet_20s import *
from nets.vgg_cifar import *
from nets.vgg_ilsvrc import *
import torch


class ModelHelper(object):
  def __init__(self, device='cuda'):
    self.device = device
    self.model_dict = {\
        'resnet_20': resnet20,\
        'resnet_32': resnet32,\
        'resnet_44': resnet44,\
        'resnet_56': resnet56,\
        'resnet_110': resnet110,\
        'resnet_1202': resnet1202,\
        'resnet_18': resnet18,\
        'resnet_34': resnet34,\
        'resnet_50': resnet50,\
        'resnet_101': resnet101,\
        'resnet_152': resnet152,\
        'vgg_16': vgg,\
        'vgg_16_ilsvrc': vgg16_bn,\
        'vgg_19_ilsvrc': vgg19_bn}

  def get_model(self, args):
    assert args.model_type in self.model_dict.keys(), 'Wrong model type.'
    # determine the number of classes
    if args.dataset == 'cifar10':
      num_classes = 10
    elif args.dataset == 'ilsvrc_12':
      num_classes = 1000

    # determine the model
    if args.model_type.startswith('resnet'):
      model = self.model_dict[args.model_type](num_classes)
    elif args.model_type.startswith('vgg'):
      if args.model_type.endswith('ilsvrc'):
        model = self.model_dict[args.model_type]()
      else:
        depth = int(args.model_type.split('_')[1])
        model = self.model_dict[args.model_type](args.dataset, depth)
    else:
      raise ValueError("Failed to get model")
    return torch.nn.DataParallel(model).cuda(self.device)


def test():
  device = torch.device('cuda:1')
  mh = ModelHelper(device)
  for k in mh.model_dict.keys():
    print(mh.get_model(k))


if __name__ == '__main__':
  test()

