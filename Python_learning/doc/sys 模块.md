``` 
import sys

print(sys.version) # 输出python的版本号
print(sys.maxsize) # 当前int能表示的最大数字
print(sys.path) # 查找模块的地址
print(sys.platform) # 输出操作系统 Linux
print(sys.argv) # 查看命令行传入的参数
```

```
sys.exit(0) # 程序退出
```

```
sys.getdefaultencoding() # 查看默认编码
sys.getfilesystemencoding() # 查看默认的文件编码
```

```
sys.getrecursionlimt(200) # 查看最大递归，大于则会报错，可以修改
```

