#!/usr/bin/env bash
# 因为联网失败后每次都要手动删除进程，这里写个脚本自动删除进程
# awk默认为空格分开，使用awk获得进程号，然后使用循环逐行读取
# & 表示开启新的进程，这样失败了不会影响主进程，wait命令等待上个进程结束后，再执行后面的代码
ps -aux | grep drcomauthsvr | awk '{print $2}' | \
  while read -r line; do echo "process ${line}"; sudo kill -9 $line & \
  wait $! ;done
sudo ./drcomauthsvr
