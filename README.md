# CableRouting
Project Homepage: https://sites.google.com/view/cablerouting/home

Data Page: https://sites.google.com/view/cablerouting/data

## Installation

#### Clone the repository
```shell
git clone git@github.com:tan-liam/CableRouting.git
cd CableRouting
```

#### Install and use the included Ananconda environment
```shell
conda create -n cable python=3.10
pip install -r requirements.txt

# CUDA 12 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# CUDA 11 installation
# Note: wheels only available on linux.
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install pytorch
```

#### Edit the following scripts to put your wandb API key into the environment variable `WANDB_API_KEY`
* `pretrain_resnet_embedding.sh`
* `train_routing_bc.sh`
* `train_highlevel.sh`
* `finetune_highlevel.sh`




## Train the model
You can train the models with the following scripts.
```shell
local_scripts/pretrain_resnet_embedding.sh
local_scripts/train_routing_bc.sh
local_scripts/train_highlevel.sh
local_scripts/finetune_highlevel.sh
```

`pretrain_resnet_embedding.sh` will use the routing data to pretrain the ResNet. Please pass the path to `route_transitions` to the `dataset_path` flag. It will output a `model.pkl` file

`train_routing_bc.sh` will train the routing policy. Please pass the path to `route_transitions` to the `dataset_path` flag. It will output a `model.pkl` file.

`train_highlevel.sh` will train the high-level policy. You will need to pass in the trained model from `pretrain_resnet_embedding.sh` to the `encoder_chekpoint_path` flag. Please pass the path to `primitive_selection_offline_dataset.npy` to the `dataset_path` flag. This will output multiple `model.pkl` files at different checkpoints.

`finetune_highlevel.sh` will fine tune the high-level policy. You will need to pass in the output from `pretrain_resnet_embedding.sh` to the `encoder_chekpoint_path` flag. You will need to pass in the output from `train_highlevel.sh` into the `primitive_policy_checkpoint_path` flag. Choose an appropriate checkpoint. Please pass in the finetuning high-level data to the `dataset_path` flag. This will output multiple `model.pkl` files at different checkpoints.

## Visualize Experiment Results with W&B
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site).
To log to W&B, you first need to set your W&B API key environment variable:
```shell
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```
Then you can run experiments with W&B logging turned on from any of the .sh scripts:
```shell
--logger.online=True
```

## Quick Check

The file `test_route.npy` is included. Use this as data for the `train_routing_bc.sh` to check that the model works as intended. We have included this for you in `train_routing_bc.sh` already.
At 500 steps the MSE should be approximately 0.426 and the eval_mse should be approximately 0.609.

## Citation BibTex

If you found this code useful, consider citing the following paper:
```
@article{luo2023multistage,
  author    = {Jianlan Luo and Charles Xu and Xinyang Geng and Gilbert Feng and Kuan Fang and Liam Tan and Stefan Schaal and Sergey Levine},
  title     = {Multi-Stage Cable Routing through Hierarchical Imitation Learning},
  journal   = {arXiv pre-print},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.08927},
}
```
