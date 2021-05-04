#!/bin/bash
date >>mount.log
df -h | grep /dev/sdc1 >>mount.log
# 如果检查到挂载了，就不执行
if test $? -ne 0; then
 sudo mount /dev/sdc1 /media/eesissi
	if [ $? -ne 0  ]; then
    		sudo umount /dev/sdc1 /media/eesissi
    		sudo mount /dev/sdc1 /media/eesissi
	fi

# 如果挂载成功，则退出码为0
    if test $? -eq 0; then echo 'mount sdc succeed'; fi
fi
