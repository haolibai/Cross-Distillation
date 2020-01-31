# Cross distillation
This repository is for our paper [Few Shot Network Compression via Cross Distillation](https://arxiv.org/abs/1911.09450), `Haoli Bai`, Jiaxiang Wu, Michael Lyu, Irwin King, AAAI 2020.

## Dependencies:
* python: 3.6+
* torch: 1.1+
* torchvision 0.2.2+
* numpy 1.14+

## Run
#### Step 1: Configuration
All the scripts to run the algorithm are in `./scripts`. Please make necessary arg changes, e.g, `data_path`, `save_path` and `load_path`. Please prepare the datasets `Cifar10` and `ImageNet` yourself.

#### Step 2: Pretrain
The alogrithm is based on a pre-trained model. For Cifar10 experiments, you can run `sh scripts/start_vanilla.sh` to train a new model from scratch. For ImageNet experiments, you can download the pretrained models from the [official website](https://pytorch.org/docs/stable/torchvision/models.html).

#### Step 3: Run
* `sh scripts/start_chnl.sh ${gpu_id}` for structured pruning
* `sh scripts/start_prune.sh ${gpu_id}` for unstructured pruning
The default parameters are already shown in the scripts. You can uncomment other configurations for different experiments.

### Tips: 
* The codes automatically generate `./few_shot_ind/`, which stores the index of sampled data for few shot training in step 3.
* Please read the arg description in `main.py` to learn more about the meanings of hyper-parameters.

## Citation
If you find the code helpful for your research, please kindly star this repo and cite our paper:
```
@inproceedings{bai2019few,
  title={Few Shot Network Compression via Cross Distillation},
  author={Bai, Haoli and Wu, Jiaxiang and King, Irwin and Lyu, Michael},
  booktitle={Proceedings of the 34-th AAAI conference on Artificial Intelligence, AAAI 2020},
  year={2020}
}
```
