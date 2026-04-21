export LOCAL_MACHINE="felipe@10.255.0.107:/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/xps_to_run_fad/frozen_vivit/"
export CHECKPOINTS_AND_INFERENCE_FODLER="/home/es119256/dados/xps/audiocraft_vivit_felipe/xps/ff71cd3f/fad"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
#rsync -avzhP --exclude='/clap' --exclude='/fad' --exclude='*.wav' --exclude='*.mp4' --exclude='*.mp3' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE
rsync -avzhP --exclude='*.bin' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE