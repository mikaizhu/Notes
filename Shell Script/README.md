# 基本操作

在开头添加注释，这样执行的时候会解析使用什么编译器运行，如下使用的bash解释器

```
#!/bin/bash
```

- 给代码添加注释

```
# comment
```

- 打印输出到终端

```
echo 'hello word'
```

将下面代码写到test文件中

```
#!/bin/bash

# this is comment

echo 'hello world'
```

> 单引号和双引号区别：单引号不会对里面的符号解析，双引号会对里面的符号解析

## 执行代码

```
bash ./test

or

./test
```

## 变量

- 定义变量，注意等号中间不能添加空格，不加引号默认是字符串

```
my_name='zhuweilin'
```

- 定义常量

```
declare -r NUM1=5
```

- 访问变量

```
在变量前加上 $ 即可访问变量里面的内容
```

- **四则运算**

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
num2=10
num3=$((NUM1+num2)) # 变量先运算再访问
num4=$((NUM1-num2))
num5=$((NUM1*num2))
num6=$((NUM1/num2))

echo "5 + 10 = $num3"
echo "5 - 10 = $num4"
echo "5 * 10 = $num5"
echo "5 / 10 = $num6"
```

- **乘方和求余数运算**

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
num2=10 

echo $((NUM1**2))
echo $((NUM1%2))
echo $((NUM1 += 2)) # 这里出错啦,改成num2就好了
```

输出结果：

```
25
1
test: line 14: NUM1: readonly variable
7
```

因为NUM1是常量，所以不会发生变化 += 就会出现问题。同样还有*=， /=

- **let命令**

let命令只能使用，看下面代码，输出什么

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
num2=2
num2+=2
echo ${num2}
```

结果为22，好像是字符串拼接

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
num2=2
let num2+=2
echo ${num2}
# 输出结果为4
```

**let 就是进行整数四则运算，如果有小数部分就会截断掉**

- **++ -- 运算**

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
rand=2
let rand+=2
echo ${rand}
echo "rand++ = $((rand++))" # 表示先打印rand，再执行++，rand=5
echo "++rand = $((++rand))" # 先加一，再打印rand，rand=6
echo "rand-- = $((rand--))" # 先打印rand，再减1
echo "--rand = $((--rand))" # 先减1，再打印rand
```

输出结果：

```
rand++ = 4
++rand = 6
rand-- = 6
--rand = 4
```

- 使用python进行运算

```
#!/bin/bash

# this is comment

num1=2.4
num2=3.3
echo "$(python -c "print(${num1} + ${num2})")"
```

使用的基本结构为

```
num1=2.4
num2=3.3
echo "$(python -c "引号里面为任何python命令")"
```

- cat文字打印

```
#!/bin/bash

# this is comment

cat<<end # 结束标志
this is 
print
many lines
end # 程序会自动识别结束标志，在这里结束，中间部分会打印到控制台
```

使用方法：cat 加两个小箭头，后面跟结束标志，可以是任何字符

## 函数

```
#!/bin/bash

# this is comment
get_date(){
date # 命令行代码
return #函数结束标志，只会返回0-255之间的数字
}

get_date #调用函数
```

然后就会输出当前日期

- **获得参数**

```
#!/bin/bash
get_sum(){
    local num3=$1;
    local num4=$2

    local sum=$((num3+num4))
    echo $sum # 函数的返回
}
num1=2
num2=4
num4=$(get_sum num1 num2)

echo "$num4"

# 正确的代码如下，为什么下面代码正确还没弄明白
#!/bin/bash
get_sum(){
    local num3=$1;
    local num4=$2

    local sum=$((num3+num4))
    echo $sum
}
num1=2
num2=4
num4=$(get_sum num1 num2)

echo "the sum is $num4"
```

> - 函数传入参数是以\$1,\$2 来接收参数的
> - 使用echo返回函数参数

## 全局变量和局部变量

```
#!/bin/bash

# this is comment
get_date(){
   name='Bob' 
   return
}
name='Alice'
get_date
echo $name
```

最后输出还是bob

将代码改成：

```
#!/bin/bash

# this is comment
get_date(){
   local name='Bob' 
   return
}
name='Alice'
get_date
echo $name
```

然后输出就是Alice了

## 从外部读取数据，使用read -p命令

```
#!/bin/bash
read -p "why commit?  " commit # commit 为变量，将输入赋值为变量
echo "the commit reason is $commit" 
```

还可以传入两个参数

```
read -p "why commit?  " commit1 commit2 # commit 为变量，将输入赋值为变量
```



## if 语句

代码如下：

```
#!/bin/bash
read -p "how old are you ? " age

if [ $age = 16 ]
then
    echo "you can drive"
elif [ $age = 15 ]
then
    echo 'you can drive next year'
else
    echo "you can't drive"
fi
```

> - If 语句以fi结尾
> - 条件要放在中括号里面



## 逻辑运算， 与或非



## if传入参数

```
#!/bin/bash
file1="./test"

if [ -e "$file1" ]; then
    echo "$file1 exists"
fi

if [ -f "$file1" ]; then
    echo "$file1 is a normal file"
fi

if [ -r "$file1" ]; then
    echo "$file1 is readable"
fi

if [ -w "$file1" ]; then
    echo "$file1 is writable"
```

> 注意中括号两边一定要留出空格子
>
> 其中还有很多参数



## 正则表达式

格式如下：

```
#!/bin/bash
read -p "is sequence? " seq
pat="^[0-9]{8}$"

if [[ $seq =~ $pat ]]; then
    echo '$seq is valid'
else
    echo "$seq not valid"
fi
```

