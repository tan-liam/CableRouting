#!/bin/bash
EXP_NAME='{INSERT NAME HERE}'
OUTPUT_DIR='{INSERT PATH HERE}'

source {INSERT PATH TO project_setup.bash}

for DECAY in 1e-2 3e-2
do
    python -m CableRouting.bc_main \
                --seed=24 \
                --dataset_path="{INSERT PATH HERE}" \
                --dataset_image_keys='wrist45_image:wrist225_image:side_image' \
                --image_augmentation='rand' \
                --total_steps=6000 \
                --eval_freq=100 \
                --batch_size=512 \
                --save_model=True \
                --lr=1e-3 \
                --weight_decay=${DECAY} \
                --policy_class_name="PretrainTanhGaussianResNetPolicy" \
                --spatial_aggregate='average' \
                --resnet_type='ResNet18' \
                --state_injection='z_only' \
                --share_resnet_between_views=False \
                --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
                --logger.online=True \
                --logger.prefix='CableRouting' \
                --logger.project="$EXP_NAME"
done