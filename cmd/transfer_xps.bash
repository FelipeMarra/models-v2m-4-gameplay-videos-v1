export LOCAL_MACHINE="felipe@10.255.0.15:/home/felipe/Desktop/checkpoints_and_inference"
export XPS_FOLDER="/app/xps/checkpoints_and_inference//app/xps/checkpoints_and_inference/d6698d5d_medium_random_10_10_25"
export CHECKPOINTS_AND_INFERENCE_FODLER="/app/xps/checkpoints_and_inference/d6698d5d_medium_random_10_10_25"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
rsync -avzhP --exclude='*.bin' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE