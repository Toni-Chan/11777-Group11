export TASK_NAME=coco_caption
export DATA_DIR=datasets
wget https://biglmdiag.blob.core.windows.net/oscar/datasets/$TASK_NAME.zip
unzip $TASK_NAME.zip -d $DATA_DIR
rm $TASK_NAME.zip
echo DONE
