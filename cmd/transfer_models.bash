export OTHER_MACHINE="felipe@10.255.0.107:/media/felipe/32740855-6a5b-4166-b047-c8177bb37be1/clap/"
#export MODEL_FODLER="/home/es119256/dados/xps/genre_classifier_new/"
export MODEL_FODLER="/home/es119256/dados/xps/clap/"

echo $OTHER_MACHINE
echo $MODEL_FODLER

rsync -avzhP --exclude='tuned*' $MODEL_FODLER $OTHER_MACHINE