export LOCAL_MACHINE="felipe@10.255.0.107:/home/felipe/Desktop/checkpoints_and_inference_orig_names"
#export XPS_FOLDER="/app/xps/checkpoints_and_inference//app/xps/"
export CHECKPOINTS_AND_INFERENCE_FODLER="/home/es119256/dados/xps/checkpoints_and_inference"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
rsync -avzhP --exclude='*.bin' $CHECKPOINTS_AND_INFERENCE_FODLER $LOCAL_MACHINE