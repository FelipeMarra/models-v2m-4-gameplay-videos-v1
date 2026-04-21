#!/bin/bash
#SBATCH --job-name=eval_txt_test          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipe.marra@ufv.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=16               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=64G                       # Memória RAM 32GB de 1007GB(Max)
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
export USER=vivit_felipe # Will create an audiocraft_felipe folder inside checkpoints

# /home/es119256/dados/xps/audiocraft_vivit_felipe/xps/ff71cd3f

XP=ff71cd3f
GAMES_LIST="legend-of-zelda-the-a-link-to-the-past, star-trek-the-next-generation-futures-past, street-fighter-ii-turbo, donkey-kong-country-3-dixie-kongs-double-trouble, tetris-2, cyber-spin, bishoujo-senshi-sailor-moon-another-story, doom, aerobiz, super-formation-soccer-ii, super-famicom-wars"

echo "Running Game-wise ImageBind Score on ViViT->MusicGen XP ${XP}"

python3 -u /home/es119256/dados/repos/visual-bardo-video/audiocraft/metrics/img_bind_consistency.py \
    --games_list "${GAMES_LIST}" \
    --eval_path /home/es119256/dados/xps/audiocraft_vivit_felipe/xps/${XP} \
    --dataset_path /home/es119256/dados/datasets/vmdb/nintendo-snes-spc

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"