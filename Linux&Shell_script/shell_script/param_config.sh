#!/bin/bash
# 使用下面方式创建Linux数组
# wine
# 脚本说明
# 此脚本为参数调节，需要在py文件中使用logging日志，以及argparse模块
wine_label_N=(5 10 12 15 20)
wine_label_K=(3 7 8 10 14)
wine_k_K=(5 10 12 15 20)
wine_k_batch_size=(10 11 10 10 11)
# mnist
mnist_label_N=(5 10 12 15 20)
mnist_label_K=(3 7 8 10 14)
mnist_k_K=(5 10 12 15 20)
mnist_k_batch_size=(20 23 22 22 23)
# 使用下面方式获得数组的长度
wine_label_len=${#wine_label_N[*]}
mnist_label_len=${#mnist_label_N[*]}
wine_k_len=${#wine_k_K[*]}
mnist_k_len=${#mnist_k_K[*]}
# 使用seq生成一个序列，类似python中的range
# 使用$()调用命令
rm *.log

echo runing wine label.py 
# 使用(())进行算数运算
for i in $(seq 0 $((${wine_label_len}-1)))
do 
  echo N:${wine_label_N[$i]}  
  echo K:${wine_label_K[$i]}  
  python3 wine_label.py \
    --n ${wine_label_N[$i]} \
    --k ${wine_label_K[$i]} 
done

echo runing wine k.py 
for i in $(seq 0 $((${wine_k_len}-1)))
do 
  echo N:${wine_k_batch_size[$i]}  
  echo K:${wine_k_K[$i]}  
  python3 wine_k.py \
    --batch_size ${wine_k_batch_size[$i]} \
    --k ${wine_k_K[$i]} 
done
######### mnist #######
echo runing mnist label.py 
for i in $(seq 0 $((${mnist_label_len}-1)))
do 
  echo N:${mnist_label_N[$i]}  
  echo K:${mnist_label_K[$i]}  
  python3 mnist_label.py \
    --n ${mnist_label_N[$i]} \
    --k ${mnist_label_K[$i]} 
done

echo runing mnist k.py 
for i in $(seq 0 $((${mnist_k_len}-1)))
do 
  echo N:${mnist_k_batch_size[$i]}  
  echo K:${mnist_k_K[$i]}  
  python3 mnist_k.py \
    --batch_size ${mnist_k_batch_size[$i]} \
    --k ${mnist_k_K[$i]} 
done
