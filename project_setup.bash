#! /bin/bash
# Project setup script
# Source this file to set up the environment for this project.


if [ ! -f ./environment.yml ]; then
    exit -1
fi
ENV_NAME=$(cat ./environment.yml | egrep "name: .+$" | sed -e 's/^name:[ \t]*//')

if [ "$1" = "setup" ]; then
    echo "Creating conda environment..."
    export CONDA_OVERRIDE_CUDA="11.8"
    conda env create -f environment.yml
elif [ "$1" = "remove" ]; then
    conda remove --name $ENV_NAME --all --yes
elif [ "$1" = "build_base" ]; then
    rm -f containers/base_img.sif
    singularity build --fakeroot containers/base_img.sif base_container.def
elif [ "$1" = "build" ]; then
    rm -f containers/code_img.sif
    singularity build --fakeroot containers/code_img.sif code_container.def
elif [ "$1" = "build_all" ]; then
    rm -f containers/base_img.sif
    singularity build --fakeroot containers/base_img.sif base_container.def
    rm -f containers/code_img.sif
    singularity build --fTakeroot containers/code_img.sif code_container.def
else
    conda activate $ENV_NAME
    export PROJECT_HOME="$(pwd)"
    export CONDA_OVERRIDE_CUDA="11.3"
    export XLA_PYTHON_CLIENT_PREALLOCATE='false'
    export PYTHONPATH="$PYTHONPATH:$PROJECT_HOME/CableRouting"
    export WANDB_API_KEY='{INSERT WANDB_API_KEY}'
fi
