set -x

source ~/miniconda3/bin/activate

python3 -m pip install 'git+https://github.com/kkoutini/passt_hear21@0.0.19#egg=hear21passt'

python3 -m pip install laion_clap

conda create --name fad python=3.9
# (/home/es119256/miniconda3/envs/fad)

conda activate fad

conda install -c conda-forge cudatoolkit=11.8.0

python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*

mkdir -p $CONDA_PREFIX/etc/conda/activate.d

echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' \
             >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' \
             >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
             
python3 -m pip install apache-beam numpy scipy tf_slim

# back to base
conda deactivate

CONDA_ENV_DIR=$(dirname $CONDA_PREFIX)

touch $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export TF_PYTHON_EXE="$CONDA_ENV_DIR/ac_eval/bin/python"' >> \
                $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
                
echo 'export TF_LIBRARY_PATH="$CONDA_ENV_DIR/ac_eval/lib/python3.10/site-packages/nvidia/cudnn/lib"' >> \
                $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh