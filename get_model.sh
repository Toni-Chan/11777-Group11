export MODEL_NAME=base-vg-labels
export MODEL_DIR=pretrained_models
sudo apt-get update
sudo apt-get install unzip
mkdir $MODEL_DIR
wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/$MODEL_NAME.zip
unzip $MODEL_NAME.zip -d $MODEL_DIR
rm $MODEL_NAME.zip
echo DONE
