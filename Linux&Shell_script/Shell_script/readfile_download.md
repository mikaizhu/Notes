需求：逐行读取文件里面的内容，然后下载对应的文件

文件内容如下：

```
file,size,link
mchar_train.zip,345.91MB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.zip
mchar_train.json,3.16MB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_train.json
mchar_val.zip,200.16MB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.zip
mchar_val.json,1.03MB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_val.json
mchar_test_a.zip,370.6MB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_test_a.zip
mchar_sample_submit_A.csv,507.83KB,http://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531795/mchar_sample_submit_A.csv
```

分别对应文件名，文件大小和下载链接

思路：使用awk命令

- 使用awk命令逐行读取
- 并以逗号为分隔
- 取出$NF即最后一个
- 传入到wget中作为标准输入，如awk "{...}" csvfile | wget $0

注意$0是标准输入，即字符串。

awk命令使用方式：

- https://www.ruanyifeng.com/blog/2018/11/awk.html
- https://opengers.github.io/linux/awk-format-print/

```
#!/bin/bash
awk -F ',' '{ print $NF }' mchar_data_list_0515.csv 
```

虽然使用上面命令，可以打印出URL，但是也会打印出link单词。


```
#!/bin/zsh
result=$(ls | grep *.csv)
awk -F ',' '/http/ {print $NF}' $result | tr -d '\r' | while read -r line;>
```

解释下这个命令的作用：

`result=$(ls | grep *.csv)` 表示将当前目录下以csv文件结尾的打印出来，并赋值给
result，因为官方给的数据文件是在csv文件中

```
awk -F ',' '/http/ {print $NF}' $result
```

awk 命令是文本处理命令，能够逐行读取文本，并进行处理

- -F ',' 表示以逗号进行分隔，否则默认以空格进行分隔，将分隔的内容依次存储进
  $0,$1,...$NF中
- /http/ {print $NF} 表示找到匹配http哪一行，然后执行打印命令。
- 因为读取的数据每一行后面都有一个回车符号，所以要使用tr命令进行去除，如果不去
  掉，使用wget命令下载，文件后面就会出现%0D
- 使用read命令逐行读取http，-r表示转义，然后执行wget命令下载

完整命令：

```
#!/bin/bash
# 读取数据下载文件
result=$(ls | grep mchar_data_list_0515.csv)

# 下载文件
awk -F ',' '/http/ {print $NF}' $result | tr -d '\r' | while read -r line; do get $line ;done

# 删除zip文件
ls | grep zip | while read line; do unzip $line ; rm $line ;done
```


