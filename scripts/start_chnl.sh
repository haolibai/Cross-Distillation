# : NOTE: vgg16-cifar10
python main.py \
  --gpu_id $1\
  --seed 0 \
  --learner chnl\
  --data_path ~/datasets/ \
  --exec_mode finetune\
  --load_path YOUR_PATH\
  --model_type vgg_16 \
  --lr 1e-5\
  --use_few_data \
  --K_shot 5 \
  --target_ratio 0.5 \
  --epochs 40 \
  --eval_epoch 10 \
  --pgd_iters 3000 \
  --pgd_lr 5e-4 \
  --use_cvx \
  --mu 0.6


# NOTE: resnet56-cifar10
# python main.py \
#   --gpu_id $1\
#   --seed 0 \
#   --learner chnl\
#   --data_path ~/datasets/ \
#   --exec_mode finetune\
#   --lr 1e-5\
#   --use_few_data \
#   --model_type resnet_56\
#   --load_path YOUR_PATH \
#   --K_shot 5 \
#   --epochs 100 \
#   --eval_epoch 20 \
#   --pgd_iters 3000 \
#   --pgd_lr 1e-3 \
#   --use_cvx \
#   --mu 0.9

# NOTE: vgg16-ilsvrc12 
# python main.py \
#   --gpu_id $1\
#   --learner chnl\
#   --dataset ilsvrc_12 \
#   --data_path ~/datasets/ilsvrc_12 \
#   --exec_mode finetune \
#   --use_few_data \
#   --lr 1e-5\
#   --model_type vgg_16_ilsvrc \
#   --load_path YOUR_PATH \
#   --no_classwise \
#   --num_data 50 \
#   --pgd_lr 5e-4 \
#   --pgd_iters 2000 \
#   --target_ratio 0.5 \
#   --use_cvx \
#   --mu 0.6 \
#   --batch_size 64 \
#   --eval_epoch 25 \
#   --epochs 100

