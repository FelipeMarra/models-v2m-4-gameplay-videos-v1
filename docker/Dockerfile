# Last release of a nvidia container of torch 2.1 since audiocraft depends on Torch 2.1
FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN apt-get update && \
    apt-get -y upgrade

RUN apt-get -y install ffmpeg

WORKDIR /app

# Audiocraft also depends on Python 3.9
# Beyond that, the docs ask for a conda env when using the metrics
# Crating a Conda env with Python 3.9

# this will ask for Enter, yes and no -> no for automatically initialize conda
# PREFIX=/root/miniconda3
RUN mkdir -p ~/miniconda3 && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_25.3.1-1-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
    rm ~/miniconda3/miniconda.sh
#source ~/miniconda3/bin/activate && conda init --all

ENV PATH=/root/miniconda3/bin:$PATH

RUN conda init --all

# Create Conda visualbardo env with Python 3.9 and Torch 2.1
# Also, by installing Ffmpeg here, I think we can skip apt-get installing it
# There is also the uninstall apex thing because of https://github.com/facebookresearch/fairseq/issues/4246#issuecomment-1799029693
RUN conda install "ffmpeg<5" -c conda-forge && \
/root/miniconda3/bin/pip install 'torch==2.1.0'

# Code will be at /app/code
# docker run \
# --gpus all \
# -d \
# -it \
# -v visualbardo_video:/root/.visualbardo_video \
# --name visualbardo_video \
# --volume /home/es119256/dados/repos/visual-bardo-video:/app/code \
# --volume /home/es119256/dados/datasets/vmdb:/app/dataset \
# --volume /home/es119256/dados/xps:/app/xps \
# --shm-size=6gb \
# visualbardo_video

# Once inside the container for the first time, run the following to install the audiocraft requirements
# python3 -m pip install -e /app/code && \
# python3 -m pip uninstall -y apex

# docker exec -it visualbardo_video sh