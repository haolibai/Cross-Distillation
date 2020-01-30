import traceback
import argparse
import torch
import torch.backends.cudnn as cudnn
from nets.model_helper import ModelHelper
from learners.basic_learner import BasicLearner
from learners.chnl_learner import ChannelLearner
from learners.weight_sparse_learner import WeightPruneLearner
from data_loader import *
from utils.helper import *
import os
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--nb_gpus', default=1, type=int, help='gpu numbers to run')
parser.add_argument('--gpu_id', default='0', help='gpu id in nvidia-smi')
parser.add_argument('--job_id', default='', help='generated automatically')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers for data loader')
parser.add_argument('--exec_mode', default='train', choices=['train', 'eval', 'finetune'])
parser.add_argument('--print_freq', default=100, type=int, help='training print frequency')
parser.add_argument('--model_type', default='', help='types of models, in nets/')
parser.add_argument('--data_path', default='~/datasets/', help='path to the dataset')
parser.add_argument('--data_aug', action='store_true', help='whether to use data augmentation')
parser.add_argument('--save_path', default='./models', help='save path for models')
parser.add_argument('--load_path', default='', help='path to load the saved model')
parser.add_argument('--eval_epoch', type=int, default=1, help='perform eval at intervals')
parser.add_argument('--use_few_data', action='store_true', help='use few data for compression or not')
parser.add_argument('--no_classwise', action='store_true', help='use classwise samples or not, e.g. N=500,1000 or K=1,2,5')
parser.add_argument('--classwise_seen_unseen', action='store_true', help='use the top num_classes for partial training.')
parser.add_argument('--num_data', default=100, type=int, help='number of smpls if no_classwise')
parser.add_argument('--K_shot', default=10, type=int, help='number of training data per class')
parser.add_argument('--distributed', action='store_true')
# learner config
parser_learner = parser.add_argument_group('Learner')
parser_learner.add_argument('--learner', default='', choices=['vanilla', 'chnl', 'weight_sparse'])
parser_learner.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
parser_learner.add_argument('--momentum', type=float, default=0.9, help='momentum value')
parser_learner.add_argument('--nesterov', action='store_true')
parser_learner.add_argument('--weight_decay', type=float, default=5e-4, help='l2 weight decay')
# dataset params
parser_dataset = parser.add_argument_group('Dataset')
parser_dataset.add_argument('--dataset', default='cifar10', choices=['cifar10', 'ilsvrc_12'], help='dataset to use')
parser_dataset.add_argument('--batch_size', type=int, default=128, help='batch_size for dataset')
parser_dataset.add_argument('--epochs', type=int, default=200, help='resnet:200, demonet:10')
# params for the proposed cross distillation method
parser_pruner = parser.add_argument_group('Pruner')
parser_pruner.add_argument('--target_ratio', type=float, default=1., help='the ratio of unpruned parameters')
parser_pruner.add_argument('--prune_first_layer', action='store_true', help='prune first layer or not?')
parser_pruner.add_argument('--pgd_once', action='store_true', help='perform pruning once at first, fast bot not accurate')
parser_pruner.add_argument('--further_augment', action='store_true', help='perform further data augmentation like mixup')
parser_pruner.add_argument('--augment_folds', type=int, default=1, help='repeat how many folds of the augmented (ensembled) data')
parser_pruner.add_argument('--use_cvx', action='store_true', \
    help='if True, use cvx combination between Lc and Li, else use soft distillation')
parser_pruner.add_argument('--alpha', type=float, default=1, help='cross distillation parameter alpha')
parser_pruner.add_argument('--beta', type=float, default=1, help='cross distillation parameter beta')
parser_pruner.add_argument('--mu', type=float, default=0.6, help='cvx combination trade off')
parser_pruner.add_argument('--pgd_iters', type=int, default=1000, help='number of iterations for pgd')
parser_pruner.add_argument('--pgd_lr', type=float, default=5e-4, help='learning rate for pgd')
parser_pruner.add_argument('--lasso_iters', type=int, default=1000, help='number of iterations for lasso, cp learner')
parser_pruner.add_argument('--lasso_lr', type=float, default=5e-4, help='learning rate for lasso, cp learner')
# * distillation config: this part is optional *
parser_distill = parser.add_argument_group('Distill')
parser_distill.add_argument('--use_kd', action='store_true')
parser_distill.add_argument('--kd_type', default='ce', choices=['ce'], help='what kind of distillation to use')
parser_distill.add_argument('--kd_temp', type=float, default=4., help='temperature for kd')
parser_distill.add_argument('--kd_regu', type=float, default=4., help='distillation loss regularizer')
parser_distill.add_argument('--kd_load_path', default='', help='load_path for the teacher model')

args = parser.parse_args()

##################### Setup ##################
# Set the seed
manualSeed = args.seed
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
# cudnn.enabled = False
cudnn.benchmark = True
cudnn.deterministic = False

# parallel setting
print("+++++++++++++++++++++++++")
print("torch version:", torch.__version__)
print("+++++++++++++++++++++++++")
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
assert len(args.gpu_id.split(',')) == args.nb_gpus, "number of gpus must match the gpu id length."
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# log setting
extra = 'few' if args.use_few_data else 'full'
if args.exec_mode in ['train']:
  args.job_id = generate_job_id()
  init_logging(os.path.join(args.save_path, '_'.join([args.model_type, args.learner, extra]), args.job_id, 'record.log'))
elif args.exec_mode == 'finetune':
  init_logging(os.path.join(os.path.dirname(args.load_path), 'ft_record.log'))
else:
  init_logging(os.path.join(os.path.dirname(args.load_path), 'record.log'))

logging.info("Using GPU: "+args.gpu_id)
print_args(vars(args))
#####################################################


def set_loader():
  if args.dataset == 'cifar10':
    return cifar10_loader(args, num_workers=4)
  elif args.dataset == 'ilsvrc_12':
    return ilsvrc12_loader(args, num_workers=4)
  else:
    raise ValueError("Unknown dataset")


def set_learner():
  if args.learner == 'vanilla':
    return BasicLearner
  elif args.learner == 'chnl':
    return ChannelLearner
  elif args.learner == 'weight_sparse':
    return WeightPruneLearner
  else:
    raise ValueError("Unknown learner")


def main():
  try:
    loaders = set_loader()
    model_helper = ModelHelper(device)
    model = model_helper.get_model(args)

    learner_fn = set_learner()
    learner = learner_fn(model, loaders, args, device)

    if args.exec_mode == 'train':
      learner.train()
    elif args.exec_mode == 'eval':
      learner.evaluate()
    elif args.exec_mode == 'finetune':
      learner.finetune()
    return 0

  except ValueError:
    traceback.print_exc()
    return 1

if __name__ == '__main__':
  main()
