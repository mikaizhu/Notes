#!/usr/bin/env bash
echo run code1
start=$(date +%s)

sleep 3 &

# 获得挂起的进程id
pid1=$!
echo run code2
sleep 5 &
pid2=$!
wait $pid1 && wait $pid2
end=$(date +%s)
run_time=$(( end-start ))
echo "run time ${run_time}s"
