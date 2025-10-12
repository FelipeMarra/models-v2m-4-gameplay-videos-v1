#!/bin/bash

export AUDIOCRAFT_TEAM=default
export USER=felipe # Will create an audiocraft_felipe folder inside checkpoints
export CUDA_VISIBLE_DEVICES=1

dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=medium \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=text2music \
    conditioners.description.t5.name=t5-base \
    conditioners.description.t5.finetune=true \
    dset=snes_mvdb \
    dataset.num_workers=6 \
    dataset.batch_size=6 \
    dataset.generate.num_samples=10 \
    dataset.valid.num_samples=500 \
    schedule.cosine.warmup=8 \
    optim.optimizer=adamw \
    optim.lr=1e-4 \
    optim.epochs=75 \
    optim.updates_per_epoch=2000 \
    optim.adam.weight_decay=0.01 \
    optim.ema.use=false \
    deadlock.timeout=1200 \
    generate.lm.prompted_samples=False \
    generate.lm.unprompted_samples=True \
    logging.log_tensorboard=true