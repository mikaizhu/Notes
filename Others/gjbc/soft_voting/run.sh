#!/usr/bin/env bash
start=$(date +%s)
echo runing dnn.py... 
python3 dnn.py \
  --train_path ../train/10type_sort_train_data_8192.npy \
  --train_label_path ../train/10type_sort_train_label_8192.npy \
  --test_path ../test/10type_sort_test_data_8192.npy \
  --test_label_path ../test/10type_sort_test_label_8192.npy \
  --val_path ../val/10type_sort_eval_data_8192.npy \
  --val_label_path ../val/10type_sort_eval_label_8192.npy \
  --epochs 50 \
  --lr 0.0001 \
  --batch_size 128 \
  --sp_start 6892\
  --sp_end 7192 \ 
  2>&1 | tee log_dnn.log &
pid1=$!

echo runing cnn.py...
python3 cnn.py \
  --train_path ../train/10type_sort_train_data_8192.npy \
  --train_label_path ../train/10type_sort_train_label_8192.npy \
  --test_path ../test/10type_sort_test_data_8192.npy \
  --test_label_path ../test/10type_sort_test_label_8192.npy \
  --val_path ../val/10type_sort_eval_data_8192.npy \
  --val_label_path ../val/10type_sort_eval_label_8192.npy \
  --epochs 50 \
  --lr 0.0001 \
  --batch_size 128 \
  --sp_start 6892 \
  --sp_end 7192 \
  2>&1 | tee log_cnn.log &
pid2=$!

wait $pid1 && wait $pid2

echo run model_stack.py...
python3 stack.py \
  --test_path ../test/10type_sort_test_data_8192.npy \
  --test_label_path ../test/10type_sort_test_label_8192.npy \
  --sp_start 6892 \
  --sp_end 7192

end=$(date +%s)
run_time=$(( end-start ))
echo "run time ${run_time}s"
