# EXP_NAME='primitive_selection_14+finetune_data0322'
# OUTPUT_DIR='/home/panda/code/cable/experiment_output/4_prim_high_level_0317'
# export XLA_PYTHON_CLIENT_PREALLOCATE=false 
# source project_setup.bash


# CMD="python -m CableRouting.primitive_selection_main"
# CMD=${CMD}" --seed=55"
# CMD=${CMD}" ----encoder_checkpoint_path=/home/panda/code/cable/experiment_output/embedding/bc_embedding_1/a44846aa51214555adc5cfca882e8ab9/model.pkl"
# CMD=${CMD}" --dataset_path=/home/panda/data/demos/multi/processed/clipa123_with_correction_5_rotated_data.npy"
# CMD=${CMD}" --image_augmentation=rand"
# CMD=${CMD}" --total_steps=30000"
# CMD=${CMD}" --batch_size=128"
# CMD=${CMD}" --eval_freq=100"
# CMD=${CMD}" --save_model=True"
# CMD=${CMD}" --lr=3e-4"
# CMD=${CMD}" --policy_class_name=TanhGaussianResNetPolicy"
# CMD=${CMD}" --weight_decay=1e-2"
# CMD=${CMD}" --policy.spatial_aggregate=average"
# CMD=${CMD}" --policy.resnet_type=ResNet18"
# CMD=${CMD}" --policy.state_injection=z_only"
# CMD=${CMD}" --policy.share_resnet_between_views=False"
# CMD=${CMD}" --logger.output_dir=${OUTPUT_DIR}/${EXP_NAME}"
# CMD=${CMD}" --logger.online=True"
# CMD=${CMD}" --logger.prefix=CableRouting"
# CMD=${CMD}" --logger.project=${EXP_NAME}"
# ${CMD} 



#! /bin/bash

EXP_NAME='{INSERT NAME HERE}'
OUTPUT_DIR='{INSERT PATH HERE}'
export XLA_PYTHON_CLIENT_PREALLOCATE=false 

source {INSERT PATH TO project_setup.bash}

XLA_PYTHON_CLIENT_PREALLOCATE=false  python -m CableRouting.primitive_selection_main \
            --seed=24 \
            --encoder_checkpoint_path="{INSERT PATH HERE}" \
            --dataset_path="{INSERT PATH HERE}" \
            --image_augmentation='rand' \
            --total_steps=30000 \
            --eval_freq=100 \
            --batch_size=128 \
            --save_model=True \
            --lr=3e-4 \
            --weight_decay=1e-2 \
            --policy.spatial_aggregate='average' \
            --policy.resnet_type='ResNet18' \
            --policy.state_injection='z_only' \
            --policy.share_resnet_between_views=False \
            --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
            --logger.online=True \
            --logger.prefix='CableRouting' \
            --logger.project="$EXP_NAME" 
 
