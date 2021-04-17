#!/bin/bash
# 挂载sdc盘的脚本
# 使用方法
# sudo chmod +x mount_sdc.sh 
# 运行脚本
# ./mount_sdc.sh
cd /media/eesissi/sdc
sudo mount /dev/sdc1 . && echo mount sdc successful!
