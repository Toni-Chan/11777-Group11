# install Ananconda to /home/ubuntu/anaconda3
# via tutorial here: https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh
bash Anaconda3-2020.07-Linux-x86_64.sh
source ~/.bashrc
conda list


# following oscar installation guide: https://github.com/microsoft/Oscar/blob/master/INSTALL.md
# create a new environment
conda create --name oscar python=3.7
conda activate oscar

# install pytorch1.2
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda

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
# git clone --recursive git@github.com:microsoft/Oscar.git
git clone https://github.com/microsoft/Oscar.git
cd Oscar
sed -i "s/git@github\.com:/https:\/\/github\.com\//" .gitmodules
git submodule update --init --recursive

cd Oscar/coco_caption
./get_stanford_models.sh
cd ..
python setup.py build develop

# install requirements
pip install -r requirements.txt
unset INSTALL_DIR


# image captioning on coc
# source : https://github.com/microsoft/Oscar/blob/master/MODEL_ZOO.md


# Fine tune train without object tags
mkdir output
python oscar/run_captioning_finetune.py \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --learning_rate 0.00003 \
    --max_seq_length 40 \
    --per_gpu_train_batch_size 64 \
    --save_steps 4000 \
    --max_steps 44000\
    --output_dir output/

# eval without object tags
python oscar/run_captioning_.py \
    --eval_model_dir output/checkpoint-14-32000 \
    --do_eval \
    --max_seq_length 40 \
    --do_lower_case


# fine tune without image features
python oscar/run_captioning_finetune.py \
    --model_name_or_path pretrained_models/base-vg-labels/ep_67_588997 \
    --do_train \
    --do_lower_case \
    --add_od_labels \
    --disable_img_features \
    --learning_rate 0.00003 \
    --per_gpu_train_batch_size 64 \
    --save_steps 4000 \
    --max_steps 44000\
    --output_dir output_img/

# eval without image features
python oscar/run_captioning_finetune.py \
    --eval_model_dir output_baseline/output/checkpoint-1-4000 \
    --do_eval \
    --do_lower_case \
    --add_od_labels \
    --disable_img_features


# inference with objectags by confidence
python oscar/run_captioning_finetune.py \
    --eval_model_dir output_baseline/output/checkpoint-14-32000 \
    --do_eval \
    --do_lower_case \
    --add_od_labels \
    --keep_top_percentage_tag_conf_threshold 0.3 \
    --keep_top_percentage_tag 0.1

# tmux command
# exit from oscar virtualenv
# start tmux session
# start oscar virtual envt
sudo apt-get install tmux
tmux new -s finetune-nolabel   finetune-noimg
tmux ls
tmux attach -t finetune-nolabel
tmux kill-session -t session-name
# C+b d to detach current tmux session

aws s3 sync output_img s3://mmml-idea1/finetune-no-objecttag
echo DONE
