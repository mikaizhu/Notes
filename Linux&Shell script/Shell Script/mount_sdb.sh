#!/bin/bash
date >>mount.log
df -h | grep /dev/sdb2 >>mount.log
# 如果检查到挂载了，就不执行
if test $? -ne 0; then
 sudo mount /dev/sdb2 /media/zwl
	if [ $? -ne 0  ]; then
    		sudo umount /dev/sdb2 /media/zwl
    		sudo mount /dev/sdb2 /media/zwl
	fi

# 如果挂载成功，则退出码为0
    if test $? -eq 0; then echo 'mount sdc succeed'; fi
fi
