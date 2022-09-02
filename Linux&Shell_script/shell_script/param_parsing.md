**参考**：https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash

**使用场景，可以对参数进行解析**：

```
./deploy.sh -t dev -u

# OR:

./deploy.sh --target dev --uglify
```

**使用方法**：

```
#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--target) target="$2"; shift ;;
        -u|--uglify) uglify="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift # 移除第二个参数
done

echo "Where to deploy: $target"
echo "Should uglify  : $uglify"
```

**用到的知识点：**

- `$@`, `$#`  表示脚本的参数数量
- `-gt` 大于 greater than
- case命令，相当于switch，case中有很多模式，`$1`表示第0个参数
- shift命令可以改变脚本参数，每次执行都会移除脚本的第一个参数，使得参数向前移动一位

**命令解析**：

- 首先判断输入的参数量是不是大于0，然后执行下面操作
- 进入case 模式 ，从第0个参数`$1`匹配下面模式。

```
# 如果第0个参数是 -t|--target 那么执行后面操作， ；表示不用换行执行后一个操作， shift表示移除这个参数，所有参数往前偏移
-t|--target) target="$2"; shift ;;
```

- 进入循环，依次读取参数

