export LOCAL_MACHINE="felipe@10.255.0.218:/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/frozen_vivit_90_10/"
export CHECKPOINTS_AND_INFERENCE_FODLER="/home/es119256/dados/xps/audiocraft_vivit_felipe_90_10/xps/cc3e7df9/fad"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
#rsync -avzhP --exclude='/clap' --exclude='/fad' --exclude='*.wav' --exclude='*.mp4' --exclude='*.mp3' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE
rsync -avzhP $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE