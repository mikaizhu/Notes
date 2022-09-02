# 可以让命令行输入python的参数

使用内置的sys模块即可

```
sys.argv[0], sys.argv[1] ...
```

在命令行中输入

```
python3 'hello'
```

注意：

```
sys.argv[0] 是文件名
sys.argv[1] 才是参数
```

