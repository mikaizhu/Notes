#!/bin/bash
for i in 1 2 3;
# 最后加个&表示前面的命令会挂起
do
  sleep ${i}s && exit 1 &
  # $!表示获得最后异步进程挂起的pid，wait等待该程序执行完后，才会继续执行后面代码
  wait $!
  # $?获得前一个程序的执行状态
  echo "${i} $?"
done
