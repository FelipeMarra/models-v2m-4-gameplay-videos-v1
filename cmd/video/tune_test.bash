#!/bin/bash

#TODO: Fine bash script to set model size, tune or not, remove previous xp or not

export AUDIOCRAFT_TEAM=default
export USER=vivit_felipe
export CUDA_VISIBLE_DEVICES=4

# /app/xps/audiocraft_felipe/ => default path

dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_video_32khz \
    model/lm/model_scale=medium \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=video2music \
    dset=snes_mvdb \
    dataset.num_workers=2 \
    dataset.batch_size=2 \
    dataset.generate.num_samples=2 \
    dataset.valid.num_samples=2 \
    schedule.cosine.warmup=1 \
    optim.optimizer=adamw \
    optim.lr=1e-4 \
    optim.epochs=2 \
    optim.updates_per_epoch=2 \
    optim.adam.weight_decay=0.01 \
    optim.ema.use=false \
    deadlock.timeout=1200 \
    generate.lm.prompted_samples=False \
    generate.lm.unprompted_samples=True \
    logging.log_tensorboard=true