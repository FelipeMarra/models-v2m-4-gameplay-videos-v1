#!/bin/bash
#SBATCH --job-name=tune_t5_visual_bardo          # Nome do job
#SBATCH --mail-type=ALL                 # Opções: BEGIN, END, FAIL, ALL, etc.
#SBATCH --mail-user=felipeferreiramarra@gmail.com       # Endereço de e-mail destinatário
#SBATCH --partition=scientific          # Partição
#SBATCH --qos=scientific-qos            # QoS 
#SBATCH --nodes=1                       # Número de nós 1 de 1
#SBATCH --ntasks=1                      # Número de tarefas
#SBATCH --cpus-per-task=8               # CPUs por tarefa 8 de 128 (Max)
#SBATCH --mem=256G                       # Memória RAM 32GB de 1007GB(Max)
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

dora -P audiocraft run \
    fsdp.use=false \
    autocast=true \
    solver=musicgen/musicgen_base_32khz \
    model/lm/model_scale=medium \
    continue_from=/app/xps/audiocraft_felipe/xps/5aa5e26a_medium_random_no_t5 \
    conditioner=text2music \
    conditioners.description.t5.name=t5-base \
    conditioners.description.t5.finetune=true \
    dset=snes_mvdb \
    dataset.num_workers=4 \
    dataset.batch_size=32 \
    +dataset.evaluate.batch_size=32 \
    +metrics.fad.tf.batch_size=32 \
    execute_only=evaluate \
    dataset.evaluate.disable_sampling=true \
    evaluate.metrics.fad=true \
    metrics.fad.use_gt=false \
    metrics.fad.tf.bin=/app/xps/fad/google-research \
    evaluate.metrics.kld=true \
    metrics.kld.use_gt=false \
    metrics.kld.passt.pretrained_length=30 \
    evaluate.metrics.genre_kld=true \
    metrics.genre_kld.use_gt=false \
    metrics.genre_kld.checkpoints=//reference/genre_classifier_new \
    evaluate.metrics.genre_class_metrics=true \
    metrics.genre_class_metrics.use_gt=false \
    metrics.genre_class_metrics.checkpoints=//reference/genre_classifier_new \
    evaluate.metrics.text_consistency=true \
    evaluate.metrics.gt_text_consistency=false \
    evaluate.metrics.tuned_text_consistency=true \
    evaluate.metrics.gt_tuned_text_consistency=false

echo "Memória final: $(free -h | grep Mem:)"
echo "Finalizado em: $(date)"