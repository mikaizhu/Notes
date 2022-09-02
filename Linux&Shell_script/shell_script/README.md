# 常见变量

- `$?`  检测上一个命令是否执行成功，因为如果命令执行成功，则输出流为0， 否则为1
- `$$`  打印当前shell的进程ID
- `$_`  上一个命令的最后一个参数
- `$!` 最近一个程序的进程ID
- `$0` 打印当前shell的名称
- `$-`  打印当前shell的启动参数
- `$@`, `$#`  表示脚本的参数数量

# first script

**场景**：每次登陆服务器，检查硬盘是否挂载，如果挂载，则不执行，没挂载则将硬盘进行挂载。

- 使用df -h判断是否挂载了硬盘
- 使用grep命令进行搜索，如果能搜到，则返回0，否则返回一个非0数字
- 使用$?输出前一个命令的返回值，检查是否运行成功

**脚本**：

```
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
```

