#!/bin/bash
# 读取数据下载文件
file=mchar_data_list_0515.csv
if test ! -f $file;then echo "$file do not exist"; exit 1; fi

result=$(ls | grep mchar_data_list_0515.csv)

# 下载文件
awk -F ',' '/http/ {print $NF}' $result | tr -d '\r' | while read -r line; do wget $line ;done

# 删除zip文件
# 因为如果解压了，scp要传很多小文件，会非常慢，直接传zip会快一点
# ls | grep zip | while read line; do unzip $line ; rm $line ;done
