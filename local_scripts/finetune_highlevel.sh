#! /bin/bash

EXP_NAME='{INSERT NAME HERE}'
OUTPUT_DIR='{INSERT PATH HERE}'
export XLA_PYTHON_CLIENT_PREALLOCATE=false 

source {INSERT PATH TO project_setup.bash}

for WEIGHT_DECAY in 1e-2
do
    XLA_PYTHON_CLIENT_PREALLOCATE=false  python -m CableRouting.primitive_selection_main \
                --seed=4876 \
                --encoder_checkpoint_path="{INSERT PATH HERE}" \
                --dataset_path="{INSERT PATH HERE}" \
                --dataset_image_keys='wrist45_image:wrist225_image:side_image' \
                --image_augmentation='rand' \
                --eval_freq=10 \
                --batch_size=128 \
                --save_model=True \
                --lr=3e-5 \
                --lr_warmup_steps=50 \
                --weight_decay=$WEIGHT_DECAY \
                --policy.spatial_aggregate='average' \
                --policy.resnet_type='ResNet18' \
                --policy.state_injection='z_only' \
                --policy.share_resnet_between_views=False \
                --logger.output_dir="$OUTPUT_DIR/$EXP_NAME" \
                --logger.online=True \
                --logger.prefix='CableRouting' \
                --logger.project="$EXP_NAME" \
                --finetune_policy=True \
                --primitive_policy_checkpoint_path='{INSERT PATH HERE}' \
                --finetune_steps=1000 \

done