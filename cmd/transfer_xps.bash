export LOCAL_MACHINE="felipe@10.255.0.107:/media/felipe/5898-0964/"
export CHECKPOINTS_AND_INFERENCE_FODLER="/home/es119256/dados/xps"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
rsync -avzhP --exclude='/clap' --exclude='/fad' --exclude='*.wav' --exclude='*.mp4' --exclude='*.mp3' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE