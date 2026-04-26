export OTHER_MACHINE="felipe@10.255.0.107:/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/ossl/"
#export MODEL_FODLER="/home/es119256/dados/xps/genre_classifier_new/"
export MODEL_FODLER="/home/es119256/dados/datasets/ossl/"

echo $OTHER_MACHINE
echo $MODEL_FODLER

rsync -avzhP $OTHER_MACHINE $MODEL_FODLER