#!/bin/bash
#SBATCH --job-name=eval_vivit_bardo_video          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipe.marra@ufv.br       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=24               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=64G                       # Memória RAM 32GB de 1007GB(Max)
#SBATCH --gres=gpu:2             # Solicitar 1 GPU de 4 (Max)
#SBATCH --time=2-00:00:00               # Tempo máximo (2 dias)
#SBATCH --output=job_%j.out        # Arquivo de saída (%j = job ID)
#SBATCH --error=job_%j.err         # Arquivo de erro

# Carregar módulos necessários
module --force purge
module load GCCcore/12.2.0 
module load CUDA/12.6.0

# Ativar ambiente
source ~/miniconda3/bin/activate
echo "$(conda info --envs)"

# Informações do job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "Memória disponível: $(free -h | grep Mem:)"
echo "Limites do processo:"
ulimit -a | egrep 'virtual memory|max resident set|open files'
echo "Iniciado em: $(date)"

set -x
set -o pipefail

# Variáveis de ambiente PyTorch
export AUDIOCRAFT_TEAM=default
export USER=vivit_felipe

export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# FAD
export CONDA_ENV_DIR="$CONDA_PREFIX/envs"
export TF_PYTHON_EXE="$CONDA_ENV_DIR/fad/bin/python"
export TF_LIBRARY_PATH="$CONDA_ENV_DIR/fad/lib/python3.10/site-packages/nvidia/cudnn/lib"

# By default dataset.evaluate.disable_sampling=true
dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_video_32khz \
    model/lm/model_scale=medium \
    continue_from=/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/b9388401 \
    conditioner=vivit2music \
    dset=snes_mvdb \
    dataset.num_workers=4 \
    dataset.batch_size=16 \
    +dataset.evaluate.batch_size=16 \
    execute_only=evaluate \
    dataset.evaluate.disable_sampling=true \
    evaluate.metrics.genre_class_metrics=true \
    metrics.genre_class_metrics.checkpoints=/home/es119256/dados/xps/genre_classifier_img_bind \
    evaluate.with_continaution=true

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"