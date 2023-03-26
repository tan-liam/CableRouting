# CableRouting
Codebase for cable routing robot.


## Installation

#### Setting up the directory structure
```shell
mkdir cable && cd cable
git clone git@github.com:tan-liam/CableRouting.git
mkdir containers
```

After doing this, you will have the following directory structure:
* cable
    * `CableRouting`: this repo
    * `base_container.def`:   singularity definition file
          for the base container, with all the dependencies installed but without the code
    * `code_container.def`:   singularity definition file
          for the code container, copying the code to base container
    * `containers`:  directory for singularity containers
    * `environment.yml`: conda environment file
    * `project_setup.bash`: project setup scripts for
          creating environment and convenient commands


Edit the following scripts to put your wandb API key into the environment variable `WANDB_API_KEY`
* `base_container.def`
* `code_container.def`
* `project_setup.bash`


#### Install and use the included Ananconda environment
```shell
./project_setup.bash setup
```



## Train the model
You can train the models with the following scripts.
```shell
local_scripts/pretrain_resnet_embedding.sh
local_scripts/train_routing_bc.sh
local_scripts/train_highlevel.sh
local_scripts/finetune_highlevel.sh
```

pretrain_resnet_embedding.sh will use the routing data to pretrain the resnet. Please pass in the routing data path to the "dataset_path" flag. It will output a model.pkl file

train_routing_bc.sh will train the routing policy. Please pass in the routing data path to the "dataset_path" flag. It will output a model.pkl file

train_highlevel.sh will train the high level policy. You will need to pass in the output from pretrain_resnet_embedding.sh to the "encoder_chekpoint_path" flag. Please pass in the high level data to the "dataset_path" flag. This will output multiple model.pkl files.

finetune_highlevel.sh will fine tune the high level policy. You will need to pass in the output from pretrain_resnet_embedding.sh to the "encoder_chekpoint_path" flag. You will need to pass in the output from train_highlevel.sh into the "primitive_policy_checkpoint_path" flag. Choose an appropriate checkpoint. Please pass in the fine tuning high level data to the "dataset_path" flag. This will output multiple model.pkl files.

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

