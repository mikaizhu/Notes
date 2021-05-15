#!/bin/bash
# conda remove -n asr --all
# conda deactivate
# 建议一行一行手动运行下面代码

env_name=asr1

conda create -n $env_name python=3.8

conda activate $env_name
require="numpy pandas matplotlib"

for i in $require
do
  conda install -c anaconda $i
done

# cuda driver and cudatookit 版本对应关系：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
# cudatookit 又叫cuda
conda install -c anaconda cudatoolkit=10.2
# 安装对应版本的pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.2 -c pytorch
# torch.__version__ 查看torch版本
# torch.version.cuda 查看环境的cuda版本
# 安装espnet
sudo apt-get install gcc
sudo apt-get install cmake
sudo apt-get install sox
sudo apt-get install libsndfile1-dev
sudo apt-get install ffmpeg
sudo apt-get install flac

cd ~/Desktop
if test ! -f espnet;then git clone https://github.com/espnet/espnet; fi
cd espnet/tools
# sudo make
CONDA_TOOLS_DIR=$(dirname ${CONDA_EXE})/..

./setup_anaconda.sh ${CONDA_TOOLS_DIR} $env_name 3.8.8
cd ~/Desktop/espnet/tools
sudo make TH_VERSION=1.8.0 CUDA_VERSION=10.2
