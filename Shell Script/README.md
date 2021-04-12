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

代码如下：

- **四则运算**

```
#!/bin/bash

# this is comment

my_name='zhuweilin'
declare -r NUM1=5
num2=10
num3=$((NUM1+num2))
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

## 全局变量和局部变量

