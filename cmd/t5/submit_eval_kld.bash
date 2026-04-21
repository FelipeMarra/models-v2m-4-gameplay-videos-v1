#!/bin/bash
#SBATCH --job-name=eval_txt          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipe.marra@ufv.br       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=16               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=32G                       # Memória RAM 32GB de 1007GB(Max)
#SBATCH --gres=gpu:1               # Solicitar 1 GPU de 4 (Max)
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

export AUDIOCRAFT_TEAM=default
export USER=felipe # Will create an audiocraft_felipe folder inside checkpoints

# FAD
export CONDA_ENV_DIR="$CONDA_PREFIX/envs"
export TF_PYTHON_EXE="$CONDA_ENV_DIR/fad/bin/python"
export TF_LIBRARY_PATH="$CONDA_ENV_DIR/fad/lib/python3.10/site-packages/nvidia/cudnn/lib"

# dados/xps/audiocraft_felipe/xps/EVAL_t5_musicgen_tuned_58b640e2/eval_gen -> fd5bcab8

dora -P audiocraft run \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=medium \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=text2music \
    conditioners.description.t5.name=t5-base \
    conditioners.description.t5.finetune=false \
    dset=snes_mvdb \
    dataset.num_workers=8 \
    dataset.batch_size=32 \
    +dataset.evaluate.batch_size=32 \
    +metrics.fad.tf.batch_size=32 \
    execute_only=evaluate \
    dataset.evaluate.disable_sampling=true \
    evaluate.metrics.fad=false \
    metrics.fad.use_gt=false \
    metrics.fad.tf.bin=/home/es119256/dados/xps/fad/google-research \
    evaluate.metrics.kld=true \
    metrics.kld.use_gt=false \
    metrics.kld.passt.pretrained_length=30 \
    evaluate.metrics.genre_class_metrics=false \
    metrics.genre_class_metrics.use_gt=false \
    metrics.genre_class_metrics.checkpoints=//reference/genre_classifier_new \
    evaluate.metrics.text_consistency=false \
    evaluate.metrics.gt_text_consistency=false \
    evaluate.metrics.save_eval_gen=true \
    evaluate.with_continaution=true

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"