#!/bin/bash
#SBATCH --job-name=tune_vivit_bardo_video          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipeferreiramarra@gmail.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=32               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=128G                       # Memória RAM 32GB de 1007GB(Max)
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

# Variáveis de ambiente PyTorch
export AUDIOCRAFT_TEAM=default
export USER=vivit_felipe # Will create an audiocraft_vivit_felipe folder inside checkpoints
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export OMP_NUM_THREADS=1

dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_video_32khz \
    model/lm/model_scale=medium \
    continue_from=//pretrained/facebook/musicgen-medium \
    conditioner=video_text2music \
    dset=snes_mvdb \
    dataset.num_workers=3 \
    dataset.batch_size=3 \
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

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"