# install Ananconda to /home/ubuntu/anaconda3
# via tutorial here: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04
cd /tml
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh


# following oscar installation guide: https://github.com/microsoft/Oscar/blob/master/INSTALL.md
# create a new environment
conda create --name oscar python=3.7
conda activate oscar

# install pytorch1.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

# install apex
export INSTALL_DIR=/home/ubuntu
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
python setup.py install --cuda_ext --cpp_ext

# configure github
# https://docs.github.com/en/free-pro-team@latest/github/authenticating-to-github/adding-a-new-ssh-key-to-your-github-account

sudo apt install unzip

# install oscar
cd $INSTALL_DIR
git clone --recursive git@github.com:microsoft/Oscar.git
cd Oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt
unset INSTALL_DIR


# image captioning on coc
# source : https://github.com/microsoft/Oscar/blob/master/MODEL_ZOO.md
mkdir output
python oscar/run_captioning.py \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --evaluate_during_training \
    --add_od_labels \
    --learning_rate 0.00003 \
    --per_gpu_train_batch_size 32 \
    --save_steps 4000 \
    --max_steps 40000\
    --output_dir output/

echo DONE
