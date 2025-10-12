#!/bin/bash

#TODO: Fine bash script to set model size, tune or not, remove previous xp or not

export AUDIOCRAFT_TEAM=default
export USER=felipe
export CUDA_VISIBLE_DEVICES=5

#rm -rf /app/xps/audiocraft_felipe/xps
#cd code

# Max batch for model=large; single GPU => 3
# Max batch for model=small; single GPU => 24
# /app/xps/audiocraft_felipe/ => default path

# single gpu, small batch size test:
dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=small \
    continue_from=//pretrained/facebook/musicgen-small \
    conditioner=text2music \
    conditioners.description.t5.name=t5-base \
    conditioners.description.t5.finetune=true \
    dset=snes_mvdb \
    dataset.num_workers=2 \
    dataset.batch_size=2 \
    dataset.train.shuffle_dataset=true \
    dataset.train.disable_sampling=true \
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