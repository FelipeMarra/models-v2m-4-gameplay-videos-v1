#!/bin/bash
#SBATCH --job-name=eval_txt_test          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipeferreiramarra@gmail.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=24               # CPUs por tarefa 8 de 128 (Max)
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
conda activate img_bind
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

python3 -u /home/es119256/dados/repos/visual-bardo-video/audiocraft/metrics/genre_acc.py \
    --checkpoints_path /home/es119256/dados/xps/genre_classifier_img_bind \
    --eval_path /home/es119256/dados/xps/audiocraft_felipe/xps/3f5557cd \
    --dataset_path /home/es119256/dados/datasets/vmdb/nintendo-snes-spc

# job 2212 eh do 29d1ea31

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"