# 基本操作

参考：https://www.runoob.com/linux/linux-shell.html

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

> - 局部变量：函数里面定义的变量
> - 环境变量：所有shell都能访问到的变量
> - shell变量：for，if else这些

## 运算


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

## 字符串操作

字符串是shell编程中最常用最有用的数据类型（除了数字和字符串，也没啥其它类型好用了），字符串可以用单引号，也可以用双引号，也可以不用引号。

**单引号**

- 单引号里的任何字符都会原样输出，单引号字符串中的变量是无效的；
- 单引号字串中不能出现单独一个的单引号（对单引号使用转义符后也不行），但可成对出现，作为字符串拼接使用。

**双引号**

- 双引号里可以有变量
- 双引号里可以出现转义字符

```
str="i am zwl\n"
echo -e $str # 要使用echo -e才会解析转译字符
```

> 字符串还有很多操作，比如读取字符串长度，注意查看教程。

## cat文字打印

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

- 参数返回，可以显示加：return 返回，如果不加，将以最后一条命令运行结果，作为返回值。 return后跟数值n(0-255)

```
#!/bin/bash

funWithReturn(){
    echo "这个函数会对输入的两个数字进行相加运算..."
    echo "输入第一个数字: "
    read aNum
    echo "输入第二个数字: "
    read anotherNum
    echo "两个数字分别为 $aNum 和 $anotherNum !"
    return $(($aNum+$anotherNum))
}
funWithReturn
echo "输入的两个数字之和为 $? !"
```

- 函数返回值在调用该函数后通过 $? 来获得。
- 注意：所有函数在使用前必须定义。这意味着必须将函数放在脚本开始部分，直至shell解释器首次发现它时，才可以使用。调用函数仅使用其函数名即可。

**获得参数**

在Shell中，调用函数时可以向其传递参数。在函数体内部，通过 $n 的形式来获取参数的值，例如，$1表示第一个参数，$2表示第二个参数...

```
funWithParam(){
    echo "第一个参数为 $1 !"
    echo "第二个参数为 $2 !"
    echo "第十个参数为 $10 !"
    echo "第十个参数为 ${10} !"
    echo "第十一个参数为 ${11} !"
    echo "参数总数有 $# 个!"
    echo "作为一个字符串输出所有参数 $* !"
}
funWithParam 1 2 3 4 5 6 7 8 9 34 73
```

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

## shell 数组

**定义数组**

bash不支持二维数组，只能使用一维数组

在 Shell 中，用括号来表示数组，数组元素用"空格"符号分割开。定义数组的一般形式为：

```
array_name=(value0 value1 value2 value3) 

or

array_name=(
value0
value1
value2
value3
)
```

**读取数组**

```
valuen=${array_name[n]}
```

看看能不能不加花括号访问到变量。 好吧，一定要花括号

**获取数组的长度**

```
# 取得数组元素的个数
length=${#array_name[@]}
# 或者
length=${#array_name[*]}
# 取得数组单个元素的长度
lengthn=${#array_name[n]}
```

## shell 参数传递

我们可以在执行 Shell 脚本时，向脚本传递参数，脚本内获取参数的格式为：**$n**。**n** 代表一个数字，1 为执行脚本的第一个参数，2 为执行脚本的第二个参数，以此类推……

以下实例我们向脚本传递三个参数，并分别输出，其中 **$0** 为执行的文件名（包含文件路径）：

```
echo "Shell 传递参数实例！";
echo "执行的文件名：$0";
echo "第一个参数为：$1";
echo "第二个参数为：$2";
echo "第三个参数为：$3";
```



```
$ chmod +x test.sh 
$ ./test.sh 1 2 3
Shell 传递参数实例！
执行的文件名：./test.sh
第一个参数为：1
第二个参数为：2
第三个参数为：3
```

**特殊字符处理**

- $#：传递到脚本的参数个数
- $$：脚本运行的当前进程ID号

> 还有其他的特殊字符，参考菜鸟教程https://www.runoob.com/linux/linux-shell-passing-arguments.html

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

## shell 基本运算符

- 算数运算符
- 关系运算符
- 布尔运算符
- 字符串运算符
- 文件测试运算符

**算术运算符**

原生bash不支持简单的数学运算，但是可以通过其他命令来实现，例如 awk 和 expr，expr 最常用。

expr 是一款表达式计算工具，使用它能完成表达式的求值操作。

```
#!/bin/bash

val=`expr 2 + 2`
echo "两数之和为 : $val"
```

> expr英文为expression，表达式计算工具
>
> 注意表达式计算使用的是反引号，不是单引号
>
> - 表达式和运算符之间要有空格，例如 2+2 是不对的，必须写成 2 + 2，这与我们熟悉的大多数编程语言不一样。
> - 完整的表达式要被 \``  包含，注意这个字符不是常用的单引号，在 Esc 键下边。

```

a=10
b=20

val=`expr $a + $b`
echo "a + b : $val"

val=`expr $a - $b`
echo "a - b : $val"

val=`expr $a \* $b`
echo "a * b : $val"

val=`expr $b / $a`
echo "b / a : $val"

val=`expr $b % $a`
echo "b % a : $val"

if [ $a == $b ]
then
   echo "a 等于 b"
fi
if [ $a != $b ]
then
   echo "a 不等于 b"
fi
```

- **注意**：条件表达式要放在方括号之间，并且要有空格，例如: \[\$a==\$b]是错误的，必须写成 \[ \$a == \$b ]。
- 乘号(*)前边必须加反斜杠(\)才能实现乘法运算；
- 在 MAC 中 shell 的 expr 语法是：**$((表达式))**，此处表达式中的 "*" 不需要转义符号 "\" 。
- 一个等号是赋值，两个等号为判断是否相等，相等则返回true

**关系运算符**

```
a=10
b=20

if [ $a -eq $b ]
then
   echo "$a -eq $b : a 等于 b"
else
   echo "$a -eq $b: a 不等于 b"
fi
if [ $a -ne $b ]
then
   echo "$a -ne $b: a 不等于 b"
else
   echo "$a -ne $b : a 等于 b"
fi
if [ $a -gt $b ]
then
   echo "$a -gt $b: a 大于 b"
else
   echo "$a -gt $b: a 不大于 b"
fi
if [ $a -lt $b ]
then
   echo "$a -lt $b: a 小于 b"
else
   echo "$a -lt $b: a 不小于 b"
fi
if [ $a -ge $b ]
then
   echo "$a -ge $b: a 大于或等于 b"
else
   echo "$a -ge $b: a 小于 b"
fi
if [ $a -le $b ]
then
   echo "$a -le $b: a 小于或等于 b"
else
   echo "$a -le $b: a 大于 b"
fi
```

输出结果：

```
10 -eq 20: a 不等于 b
10 -ne 20: a 不等于 b
10 -gt 20: a 不大于 b
10 -lt 20: a 小于 b
10 -ge 20: a 小于 b
10 -le 20: a 小于或等于 b
```

**布尔运算符**

布尔运算主要是与或非运算

- ！非运算：非运算，表达式为 true 则返回 false，否则返回 true。`[ ! false ] 返回 true。`
- \-o 或运算：或运算，有一个表达式为 true 则返回 true。`[ $a -lt 20 -o $b -gt 100 ] 返回 true。`
- \-a 与运算：与运算，两个表达式都为 true 才返回 true。`[ $a -lt 20 -a $b -gt 100 ] 返回 false。`

**逻辑运算符**

- \&&:逻辑的 AND, `[[ $a -lt 100 && $b -gt 100 ]] 返回 false`
- \||:逻辑的 OR, `[[ $a -lt 100 || $b -gt 100 ]] 返回 true`

```
a=10
b=20

if [[ $a -lt 100 && $b -gt 100 ]]
then
   echo "返回 true"
else
   echo "返回 false"
fi

if [[ $a -lt 100 || $b -gt 100 ]]
then
   echo "返回 true"
else
   echo "返回 false"
fi
```

```
返回 false
返回 true
```

**字符串运算符**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gphwcho7upj312g0ek0uw.jpg" alt="image.png" style="zoom:50%;" />

```
a="abc"
b="efg"

if [ $a = $b ]
then
   echo "$a = $b : a 等于 b"
else
   echo "$a = $b: a 不等于 b"
fi
if [ $a != $b ]
then
   echo "$a != $b : a 不等于 b"
else
   echo "$a != $b: a 等于 b"
fi
if [ -z $a ]
then
   echo "-z $a : 字符串长度为 0"
else
   echo "-z $a : 字符串长度不为 0"
fi
if [ -n "$a" ]
then
   echo "-n $a : 字符串长度不为 0"
else
   echo "-n $a : 字符串长度为 0"
fi
if [ $a ]
then
   echo "$a : 字符串不为空"
else
   echo "$a : 字符串为空"
fi
```

```
abc = efg: a 不等于 b
abc != efg : a 不等于 b
-z abc : 字符串长度不为 0
-n abc : 字符串长度不为 0
abc : 字符串不为空
```

**文件测试运算符**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gphweb884lj312k0xawkb.jpg" alt="image.png" style="zoom:50%;" />

```
file="/var/www/runoob/test.sh"
if [ -r $file ]
then
   echo "文件可读"
else
   echo "文件不可读"
fi
if [ -w $file ]
then
   echo "文件可写"
else
   echo "文件不可写"
fi
if [ -x $file ]
then
   echo "文件可执行"
else
   echo "文件不可执行"
fi
if [ -f $file ]
then
   echo "文件为普通文件"
else
   echo "文件为特殊文件"
fi
if [ -d $file ]
then
   echo "文件是个目录"
else
   echo "文件不是个目录"
fi
if [ -s $file ]
then
   echo "文件不为空"
else
   echo "文件为空"
fi
if [ -e $file ]
then
   echo "文件存在"
else
   echo "文件不存在"
fi
```

```
文件可读
文件可写
文件可执行
文件为普通文件
文件不是个目录
文件不为空
文件存在
```

## Printf 命令

printf 使用引用文本或空格分隔的参数，外面可以在 printf 中使用格式化字符串，还可以制定字符串的宽度、左右对齐方式等。默认 printf 不会像 echo 自动添加换行符，我们可以手动添加 \n。

```
printf  format-string  [arguments...]
```

**参数说明：**

- **format-string:** 为格式控制字符串
- **arguments:** 为参数列表。

```
$ echo "Hello, Shell"
Hello, Shell
$ printf "Hello, Shell\n"
Hello, Shell
$
```

```
printf "%-10s %-8s %-4s\n" 姓名 性别 体重kg  
printf "%-10s %-8s %-4.2f\n" 郭靖 男 66.1234
printf "%-10s %-8s %-4.2f\n" 杨过 男 48.6543
printf "%-10s %-8s %-4.2f\n" 郭芙 女 47.9876
```

```
姓名     性别   体重kg
郭靖     男      66.12
杨过     男      48.65
郭芙     女      47.99
```

- **%s %c %d %f** 都是格式替代符，**％s** 输出一个字符串，**％d** 整型输出，**％c** 输出一个字符，**％f** 输出实数，以小数形式输出。
- **%-10s** 指一个宽度为 10 个字符（**-** 表示左对齐，没有则表示右对齐），任何字符都会被显示在 10 个字符宽的字符内，如果不足则自动以空格填充，超过也会将内容全部显示出来。
- **%-4.2f** 指格式化为小数，其中 **.2** 指保留2位小数。

**printf转义序列**

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gphwwmepl0j312k0s20vt.jpg)

## shell test命令

Shell中的 test 命令用于检查某个条件是否成立，它可以进行数值、字符和文件三个方面的测试。

**数值测试**

```
num1=100
num2=100
if test $[num1] -eq $[num2]
then
    echo '两个数相等！'
else
    echo '两个数不相等！'
fi
```

**字符串测试**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gphx1a9w27j31140aa750.jpg" alt="image.png" style="zoom:50%;" />

```
num1="ru1noob"
num2="runoob"
if test $num1 = $num2
then
    echo '两个字符串相等!'
else
    echo '两个字符串不相等!'
fi
```

**文件测试**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gphx24u437j30ws0lataz.jpg" alt="image.png" style="zoom:50%;" />

```
cd /bin
if test -e ./bash
then
    echo '文件已存在!'
else
    echo '文件不存在!'
fi
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

## shell控制流 if语句 for语句等

**if语句语法**

```
if condition
then
    command1 
    command2
    ...
    commandN 
fi
```

写成一行（适用于终端命令提示符）：

```
if [ $(ps -ef | grep -c "ssh") -gt 1 ]; then echo "true"; fi
```

**if else 语法**

```
if condition
then
    command1 
    command2
    ...
    commandN
else
    command
fi
```

**if elif else语法**

```
if condition1
then
    command1
elif condition2 
then 
    command2
else
    commandN
fi
```

if else 语句经常与 test 命令结合使用，如下所示：

```
num1=$[2*3]
num2=$[1+5]
if test $[num1] -eq $[num2]
then
    echo '两个数字相等!'
else
    echo '两个数字不相等!'
fi
```



 **if传入参数**

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

### **for 语句**

```
for var in item1 item2 ... itemN
do
    command1
    command2
    ...
    commandN
done
```

写成一行：

```
for var in item1 item2 ... itemN; do command1; command2… done;
```

当变量值在列表里，for 循环即执行一次所有命令，使用变量名获取列表中的当前取值。命令可为任何有效的 shell 命令和语句。in 列表可以包含替换、字符串和文件名。

in列表是可选的，如果不用它，for循环使用命令行的位置参数。

例如，顺序输出当前列表中的数字：

```
for loop in 1 2 3 4 5
do
    echo "The value is: $loop"
done
```

顺序输出字符串中的字符：

```
#!/bin/bash

for str in This is a string
do
    echo $str
done
```

### while 语句

```
while condition
do
    command
done
```

```
#!/bin/bash
int=1
while(( $int<=5 ))
do
    echo $int
    let "int++"
done
```

while循环可用于读取键盘信息。下面的例子中，输入信息被设置为变量FILM，按<Ctrl-D>结束循环。

```
echo '按下 <CTRL-D> 退出'
echo -n '输入你最喜欢的网站名: '
while read FILM
do
    echo "是的！$FILM 是一个好网站"
done
```

无限循环

```
while true
do
    command
done
```

### Until 循环

until 循环执行一系列命令直至条件为 true 时停止。

until 循环与 while 循环在处理方式上刚好相反。

一般 while 循环优于 until 循环，但在某些时候—也只是极少数情况下，until 循环更加有用。

until 语法格式:

```
until condition
do
    command
done
```

condition 一般为条件表达式，如果返回值为 false，则继续执行循环体内的语句，否则跳出循环。

以下实例我们使用 until 命令来输出 0 ~ 9 的数字：

```
a=0

until [ ! $a -lt 10 ]
do
   echo $a
   a=`expr $a + 1`
done
```

### Case esac

**case ... esac** 为多选择语句，与其他语言中的 switch ... case 语句类似，是一种多分枝选择结构，每个 case 分支用右圆括号开始，用两个分号 **;;** 表示 break，即执行结束，跳出整个 case ... esac 语句，esac（就是 case 反过来）作为结束标记。

可以用 case 语句匹配一个值与一个模式，如果匹配成功，执行相匹配的命令。

**case ... esac** 语法格式如下：

```
case 值 in
模式1)
    command1
    command2
    ...
    commandN
    ;;
模式2）
    command1
    command2
    ...
    commandN
    ;;
esac
```

case 工作方式如上所示，取值后面必须为单词 **in**，每一模式必须以右括号结束。取值可以为变量或常数，匹配发现取值符合某一模式后，其间所有命令开始执行直至 **;;**。

取值将检测匹配的每一个模式。一旦模式匹配，则执行完匹配模式相应命令后不再继续其他模式。如果无一匹配模式，使用星号 * 捕获该值，再执行后面的命令。

下面的脚本提示输入 1 到 4，与每一种模式进行匹配：

```
echo '输入 1 到 4 之间的数字:'
echo '你输入的数字为:'
read aNum
case $aNum in
    1)  echo '你选择了 1'
    ;;
    2)  echo '你选择了 2'
    ;;
    3)  echo '你选择了 3'
    ;;
    4)  echo '你选择了 4'
    ;;
    *)  echo '你没有输入 1 到 4 之间的数字'
    ;;
esac
```

### 跳出循环

在循环过程中，有时候需要在未达到循环结束条件时强制跳出循环，Shell使用两个命令来实现该功能：break和continue。

break命令允许跳出所有循环（终止执行后面的所有循环）。

```
#!/bin/bash
while :
do
    echo -n "输入 1 到 5 之间的数字:"
    read aNum
    case $aNum in
        1|2|3|4|5) echo "你输入的数字为 $aNum!"
        ;;
        *) echo "你输入的数字不是 1 到 5 之间的! 游戏结束"
            break
        ;;
    esac
done
```

continue命令与break命令类似，只有一点差别，它不会跳出所有循环，仅仅跳出当前循环。

## shell输入输出重定向

参考：https://www.runoob.com/linux/linux-shell-io-redirections.html

## shell文件包含

参考：https://www.runoob.com/linux/linux-shell-include-file.html

和其他语言一样，Shell 也可以包含外部脚本。这样可以很方便的封装一些公用的代码作为一个独立的文件。

Shell 文件包含的语法格式如下：

```
. filename   # 注意点号(.)和文件名中间有一空格

或

source filename
```

**实例：**

创建两个shell文件

Test1.sh 代码如下：

```
#!/bin/bash
url="http://www.runoob.com"
```

Test2.sh代码如下：

```
#!/bin/bash
#使用 . 号来引用test1.sh 文件
. ./test1.sh
# 或者使用以下包含文件代码
# source ./test1.sh
echo "菜鸟教程官网地址：$url"
```



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

