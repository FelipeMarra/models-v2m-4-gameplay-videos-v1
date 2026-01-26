#!/bin/bash

# Carregar módulos necessários
module --force purge
module load Apptainer/1.2.2

# /tmp
export APPTAINER_TMPDIR="/home/es119256/tmp/"
export SINGULARITY_TMPDIR="/home/es119256/tmp/"
# /cache
export SINGULARITY_CACHEDIR="/home/es119256/cache/"
export APPTAINER_CACHEDIR="/home/es119256/cache/"

SING_RECIPE=/home/es119256/dados/repos/visual-bardo-video/docker/visual_bardo_video.def
SING_IMAGE=/home/es119256/dados/repos/visual-bardo-video/docker/visual_bardo_video.sif

singularity build $SING_IMAGE $SING_RECIPE