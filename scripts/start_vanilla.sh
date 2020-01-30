python main.py \
  --nb_gpus 1\
  --gpu_id 0\
  --momentum 0.9\
  --exec_mode train\
  --learner vanilla\
  --dataset cifar10\
  --data_path ~/datasets/ \
  --model_type vgg_16 \
  --epochs 300\
  --batch_size 64 \
  --lr 1e-1\
  --weight_decay 5e-4

