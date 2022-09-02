# crontab定时执行代码

## 常用命令

参考网址：https://blog.csdn.net/u013383813/article/details/74741004

**编辑crontab文件：**

```
crontab -e # e表示edit
```

输入上面这个命令之后，就可以进行编辑任务了，选择vim进行编辑

**查看crontab的任务：**

```
crontab -l # l表示list
```

**重启任务：**

```
service cron restart
```

**查看cron的状态**

```
service cron status
```

# crontab说明

 **每个位置的时间说明：**

```
minute hour day month week command
```

- minute范围在0-59
- hour在0-23
- day 在1-31
- week 在0-7

**常用符号说明：**

- 每天，或者每小时。表示任意的意思

```
*
```

如：

```
* * * * * date >> ~/Desktop/time.log
```

表示每月，每天，每小时，每分钟，查看一次时间

- 逗号表示指定多个时间

```
，
```

如：

```
20,40 * * * * date >> ~/Desktop/time.log
```

表示每小时的20分和40分的时候，显示一次时间

- 减号，表示一个整数范围

```
-
```

如：

```
* 2-5 * * * date >> ~/Desktop/time.log
```

表示每天的2点到5点，每分钟都显示时间

- 除号，表示时间间隔

```
/
```

如：

```
*/15 * * * * date >> ~/Desktop/time.log
```

表示每隔15分钟就检测一次时间

```
0-59/15 * * * * date >> ~/Desktop/time.log
```

同理

## 执行python文件

- 不同行的任务，是同时执行的，也就是只要到了时间，就会执行，不需要等前面命令执行完了

因为crontab的任务，默认是在bash中执行的，

**所以下面这样是不能执行的：**

```
* * * * * python3 test1.py
```

**只要告诉用什么python，并且告诉文件在哪里就行：**

如何使用想用的python来运行文件？使用下面命令即可，使用miniconda 中的python3

```
whereis python3
```

如果没有显示miniconda的环境，要先在外部激活conda环境，再输入whereis

```
conda activate python38_env
```

如：

```
* * * * * /home/zwl/miniconda3/python38_env/bin/python3.8 ~/Desktop/test1.py
```

这样写路径太麻烦了，可以将命令和起来写

```
* * * * * cd ~/Desktop && ....python3 test1.py
```

**如服务器上要运行python代码：**

```
50 23 * * * cd /home/zwl/Desktop/wireless/DNN && /home/zwl/miniconda3/bin/python3 dnn1.py 32 60
50 23 * * * cd /home/zwl/Desktop/wireless/DNN && /home/zwl/miniconda3/bin/python3 dnn2.py 32 60
50 23 * * * cd /home/zwl/Desktop/wireless/DNN && /home/zwl/miniconda3/bin/python3 dnn3.py 32 60
59 23 * * * cd /home/zwl/Desktop/wireless/DNN && /home/zwl/miniconda3/bin/python3 dnn4.py 32 60
```

