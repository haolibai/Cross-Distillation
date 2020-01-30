# NOTE: vgg16 on cifar10
python main.py \
  --gpu_id $1\
  --seed 1\
  --learner weight_sparse \
  --exec_mode finetune \
  --lr 1e-3\
  --use_few_data \
  --K_shot 5 \
  --target_ratio 0.1 \
  --model_type vgg_16 \
  --load_path YOUR_PATH \
  --eval_epoch 20 \
  --epochs 100 \
  --pgd_iters 3000 \
  --pgd_lr 5e-4 \
  --use_cvx \
  --mu 0.6

# NOTE: resnet on Imagenet
# python main.py \
#   --gpu_id $1 \
#   --seed 0\
#   --learner weight_sparse\
#   --use_few_data \
#   --exec_mode finetune \
#   --data_path /data1/hlbai/datasets/ilsvrc_12/ \
#   --dataset ilsvrc_12 \
#   --model_type resnet_34 \
#   --load_path YOUR_PATH \ 
#   --no_classwise \
#   --classwise_seen_unseen \
#   --num_data 50 \
#   --lr 1e-5 \
#   --pgd_iters 2000 \
#   --pgd_lr 5e-4 \
#   --batch_size 64 \
#   --target_ratio 0.1 \
#   --eval_epoch 50 \
#   --epochs 100 \
#   --use_cvx \
#   --mu 0.6

