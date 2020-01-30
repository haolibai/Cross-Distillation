import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Subset
import pickle
import pdb
import numpy as np


### Functions for the few shot indices sampler ###
def classwise_get_indices(dataset_name, dataset, K_shot, num_class, seed, seen_unseen=False):
  file_dir = './few_shot_ind'
  file_name = os.path.join(file_dir, \
      '_'.join([dataset_name, 'k'+str(K_shot), 'c'+str(num_class), 's'+str(seed)+'.pkl']))
  if seen_unseen:
    file_name = file_name[:-4] + '_sus.pkl'

  try:
    with open(file_name, 'rb') as f:
      ind = pickle.load(f)
      print("Indices restored from %s" % file_name)
  except:
    if not os.path.isdir(file_dir):
      os.makedirs(file_dir)
    with open(file_name, 'wb') as f:
      print("Searching for new indices for %s" % file_name)
      ind = classwise_search_indices(dataset, K_shot, num_class, seed)
      pickle.dump(ind, f)
      print("Indices stored at %s" % file_name)
  return ind


def classwise_search_indices(dataset, K_shot, num_class, seed):
  """ Used for searching indices. Could be slow for the first time, one instance a time """
  indices = []
  nb_smpls_left = [K_shot] * num_class
  list_iters = list(range(len(dataset)))
  if seed != 0:
    np.random.seed(seed)
    np.random.shuffle(list_iters)
  for itr in list_iters:
    x, y = dataset[itr]
    if nb_smpls_left[y] > 0:
      indices.append(itr)
      nb_smpls_left[y] -= 1

    done = True
    for c in range(num_class):
      done = (nb_smpls_left[c] == 0 and done)
    if done:
      break
  return indices


def get_indices(dataset, num_instances, seed):
  indices = search_indices(dataset, num_instances, seed)
  print("Using classwisely independent sampling..., %d instances in total" % num_instances)
  return indices


def search_indices(dataset, num_instances, seed):
  """ Used for searching indices. Could be slow for the first time, one instance a time """
  list_iters = list(range(len(dataset)))
  np.random.seed(seed)
  np.random.shuffle(list_iters)
  indices = list_iters[:num_instances]
  return indices


class pseudo_loader(object):
  """ load all the few shot images in memory and index them accordingly """

  def __init__(self, loader, batch_size, total_num, shuffle=False):
    self.total_num = total_num
    self.xs, self.ys = next(iter(loader)) # load out all the images
    self.batch_size = min(batch_size, self.total_num)
    self.shuffle = shuffle
    self.counter = 0

  def __next__(self):
    # NOTE: a self-infinitely loop function
    if self.shuffle:
      shuffle_ind = np.arange(self.total_num)
      np.random.shuffle(shuffle_ind)
      self.xs, self.ys = self.xs[shuffle_ind], self.ys[shuffle_ind]

    if self.counter + self.batch_size <= self.total_num:
      xs = self.xs[self.counter:self.counter+self.batch_size]
      ys = self.ys[self.counter:self.counter+self.batch_size]
      self.counter += self.batch_size
    else:
       xs = self.xs[self.counter:]
       ys = self.ys[self.counter:]
       self.counter = self.batch_size - (self.total_num - self.counter)
       xs = torch.cat((xs, self.xs[:self.counter]), dim=0)
       ys = torch.cat((ys, self.ys[:self.counter]), dim=0)
    return xs, ys

  def __iter__(self):
    return self


def cifar10_loader(args, num_workers=4):
  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                               std=[x/255.0 for x in [63.0, 62.1, 66.7]])
  if args.data_aug:
    with torch.no_grad():
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda x: F.pad(
                                          Variable(x.unsqueeze(0), requires_grad=False),
                                          (4,4,4,4),mode='reflect').data.squeeze()),
      transforms.ToPILImage(),
      transforms.RandomCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      ])
      transform_test = transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
  else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
  trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
  testset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform_test)
  if args.use_few_data:
    indices = classwise_get_indices(args.dataset, trainset, args.K_shot, 10, args.seed)
    trainset = Subset(trainset, indices=indices)
    few_shot_num = args.K_shot * 10
    train_loader = DataLoader(trainset, batch_size=few_shot_num, shuffle=False, num_workers=num_workers, pin_memory=True)
    train_loader = pseudo_loader(train_loader, args.batch_size, few_shot_num, shuffle=False)
    # train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    # train_loader = iter(cycle(train_loader))
  else:
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  return train_loader, test_loader


def ilsvrc12_loader(args, num_workers=4):
  assert not (args.use_few_data and args.distributed), \
      'when using few data for layerwise operation, do not use distirbuted'

  traindir = os.path.join(args.data_path, 'train')
  valdir = os.path.join(args.data_path, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]))

  if args.use_few_data:
    print('Using few shot data.')
    if not args.no_classwise:
      indices = classwise_get_indices(args.dataset, train_dataset, args.K_shot, 1000, args.seed)
    else:
      if not args.classwise_seen_unseen:
        indices = get_indices(train_dataset, args.num_data, args.seed)
      else:
        # search index for seen class. 1-shot
        indices = classwise_get_indices(args.dataset, train_dataset, 1, args.num_data, args.seed, args.classwise_seen_unseen)

    #in ilsvrc_12, loader will continuously read data
    train_dataset = Subset(train_dataset, indices=indices)

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
    train_sampler = None

  if args.use_few_data:
    if not args.no_classwise:
      args.num_data = args.K_shot * 1000
      print("num of training instances (classwisely): %d" % args.num_data)
    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.num_data, shuffle=(train_sampler is None),
      num_workers=num_workers, pin_memory=True, sampler=train_sampler)
    train_loader = pseudo_loader(train_loader, args.batch_size, args.num_data, shuffle=False)
    # train_loader = iter(cycle(train_loader))
  else:
    train_loader = torch.utils.data.DataLoader(
      train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
      num_workers=num_workers, pin_memory=True, sampler=train_sampler)

  val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)

  return train_loader, val_loader


