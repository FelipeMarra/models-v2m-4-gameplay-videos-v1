#!/bin/bash
#SBATCH --job-name=tune_vivit_bardo_video          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipeferreiramarra@gmail.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=64               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=1007GB                       # Memória RAM 32GB de 1007GB(Max)
#SBATCH --gres=gpu:3                 # Solicitar 1 GPU de 4 (Max)
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
echo "Iniciado em: $(date)"

# Executar seu programa
export AUDIOCRAFT_TEAM=default
export USER=vivit_felipe # Will create an audiocraft_vivit_felipe folder inside checkpoints

dora -P audiocraft run -d \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_video_32khz \
    model/lm/model_scale=medium \
    conditioner=video2music \
    dset=snes_mvdb \
    dataset.num_workers=6 \
    dataset.batch_size=6 \
    dataset.train.shuffle=true \
    dataset.train.disable_sampling=true \
    dataset.generate.num_samples=10 \
    dataset.valid.num_samples=500 \
    schedule.cosine.warmup=8 \
    optim.optimizer=adamw \
    optim.lr=1e-4 \
    optim.epochs=150 \
    optim.updates_per_epoch=null \
    optim.adam.weight_decay=0.01 \
    optim.ema.use=false \
    deadlock.timeout=1200 \
    generate.lm.prompted_samples=False \
    generate.lm.unprompted_samples=True \
    logging.log_tensorboard=true

echo "Finalizado em: $(date)"
