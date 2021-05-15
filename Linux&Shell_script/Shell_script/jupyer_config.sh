#!/bin/bash
# 安装jupyter 拓展插件
pip install jupyter_nbextensions_configurator
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# 安装jupyter vim
# 网站：https://github.com/lambdalisue/jupyter-vim-binding
# Create required directory in case (optional)
mkdir -p $(jupyter --data-dir)/nbextensions
# Clone the repository
cd $(jupyter --data-dir)/nbextensions
git clone https://github.com/lambdalisue/jupyter-vim-binding vim_binding
# Activate the extension
jupyter nbextension enable vim_binding/vim_binding
