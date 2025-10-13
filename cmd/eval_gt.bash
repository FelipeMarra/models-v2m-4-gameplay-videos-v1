#!/bin/bash

export AUDIOCRAFT_TEAM=default
export USER=vivit_felipe
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

# FAD
export CONDA_ENV_DIR="$CONDA_PREFIX/envs"
export TF_PYTHON_EXE="$CONDA_ENV_DIR/fad/bin/python"
export TF_LIBRARY_PATH="$CONDA_ENV_DIR/fad/lib/python3.10/site-packages/nvidia/cudnn/lib"

# By default dataset.evaluate.disable_sampling=true
dora -P audiocraft run \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small\
    conditioner=text2music \
    conditioners.description.t5.name=t5-base \
    conditioners.description.t5.finetune=true \
    dset=snes_mvdb \
    dataset.num_workers=4 \
    dataset.batch_size=32 \
    +dataset.evaluate.batch_size=32 \
    +metrics.fad.tf.batch_size=32 \
    execute_only=evaluate \
    dataset.evaluate.disable_sampling=true \
    evaluate.metrics.fad=true \
    metrics.fad.use_gt=true \
    metrics.fad.tf.bin=/app/xps/fad/google-research \
    evaluate.metrics.kld=true \
    metrics.kld.use_gt=true \
    metrics.kld.passt.pretrained_length=30 \
    evaluate.metrics.genre_kld=true \
    metrics.genre_kld.use_gt=true \
    metrics.genre_kld.checkpoints=//reference/genre_classifier_new \
    evaluate.metrics.genre_class_metrics=true \
    metrics.genre_class_metrics.use_gt=true \
    metrics.genre_class_metrics.checkpoints=//reference/genre_classifier_new \
    evaluate.metrics.text_consistency=false \
    evaluate.metrics.gt_text_consistency=true \
    evaluate.metrics.tuned_text_consistency=false \
    evaluate.metrics.gt_tuned_text_consistency=true