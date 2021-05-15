#!/bin/bash
#conda remove -n asr --all
env_name=asr

conda create -n $env_name python=3.7.9
require="numpy pandas matplotlib opencv"
for i in $require
do
  conda install -c anaconda $i
done

# 安装对应版本的pytorch
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

# 安装espnet
sudo apt-get install cmake
sudo apt-get install sox
sudo apt-get install libsndfile1-dev
sudo apt-get install ffmpeg
sudo apt-get install flac

cd ~/Desktop
git clone https://github.com/espnet/espnet
cd espnet/tools
sudo make
CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..

./setup_anaconda.sh ${CONDA_TOOLS_DIR} $env_name 3.7.9
cd ~/Desktop/espnet/tools
make TH_VERSION=1.3.1 CUDA_VERSION=10.1
