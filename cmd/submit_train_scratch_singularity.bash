#!/bin/bash
#SBATCH --job-name=tune_vivit_bardo_video          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipeferreiramarra@gmail.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=64               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=500GB                       # Memória RAM 32GB de 1007GB(Max)
#SBATCH --gres=gpu:1                  # Solicitar 1 GPU de 4 (Max)
#SBATCH --time=2-00:00:00               # Tempo máximo (2 dias)
#SBATCH --output=job_%j.out        # Arquivo de saída (%j = job ID)
#SBATCH --error=job_%j.err         # Arquivo de erro

# Informações do job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "Iniciado em: $(date)"

# Carregar módulos necessários
module --force purge
module load GCCcore/12.2.0 
module load CUDA/12.6.0
module load singularity-bindings/custom-1.0

# Run Singularity image
SING_IMG=/home/es119256/dados/repos/visual-bardo-video/docker/visual_bardo_video.sif

singularity exec $SING_IMG

echo "Finalizado em: $(date)"
