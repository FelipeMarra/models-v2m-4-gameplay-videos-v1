# This was used to calculate FAD on my personal computer. May come in handy again since FAD it doesn't like slurm very much

set -xeuo -pipefail

FAD_MODEL_PATH="/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/vggish_model.ckpt"
FAD_ENV_PATH="/home/felipe/anaconda3/envs/fad"

export TF_PYTHON_EXE="$FAD_ENV_PATH/bin/python"
export TF_LIBRARY_PATH="$FAD_ENV_PATH/lib/python3.10/site-packages/nvidia/cudnn/lib"
export PYTHONPATH="/home/felipe/Documents/Github/google-research-fad" 

FILES_TEST="/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/tuned_vit/fad/files_tests.cvs"
STATS_TESTS="/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/tuned_vit/fad/stats_tests"

FILES_BACK="/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/tuned_vit/fad/files_background.cvs"
STATS_BACK="/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/tuned_vit/fad/stats_background"

CALC_TEST="$TF_PYTHON_EXE -m frechet_audio_distance.create_embeddings_main --model_ckpt $FAD_MODEL_PATH --input_files $FILES_TEST --stats $STATS_TESTS --batch_size 1"
CALC_BACK="$TF_PYTHON_EXE -m frechet_audio_distance.create_embeddings_main --model_ckpt $FAD_MODEL_PATH --input_files $FILES_BACK --stats $STATS_BACK --batch_size 1"
CALC_FAD="$TF_PYTHON_EXE -m frechet_audio_distance.compute_fad --test_stats $STATS_TESTS --background_stats $STATS_BACK"

source /root/miniconda3/bin/activate 
conda activate fad 

echo "CALC TEST"
${CALC_TEST} 
echo "CALC TEST COMPLETED"

echo "CALC BACK"
${CALC_BACK}
echo "CALC BACK COMPLETED"

echo "CALC FAD"
${CALC_FAD}
echo "CALC FAD COMPLETED"
