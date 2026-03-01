export LOCAL_MACHINE="felipe@10.255.0.107:/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/ossl_vivit_pt/"
#export XPS_FOLDER="/app/xps/checkpoints_and_inference//app/xps/"
export CHECKPOINTS_AND_INFERENCE_FODLER="/home/es119256/dados/datasets/oss_test_pt"

#rsync -avzhP --exclude='*.th' $XPS_FOLDER $LOCAL_MACHINE
#rsync -avzhP $XPS_FOLDER $LOCAL_MACHINE

echo $CHECKPOINTS_AND_INFERENCE_FODLER
echo $LOCAL_MACHINE
rsync -avzhP $LOCAL_MACHINE $CHECKPOINTS_AND_INFERENCE_FODLER