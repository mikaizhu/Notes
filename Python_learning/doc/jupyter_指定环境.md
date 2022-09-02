有时候直接输入

```
jupyter notebook
```

`import sys;sys.executable` 查看环境，发现使用的并不是想要的环境

解决办法：

```
# 直接使用指定的conda环境路径，来运行jupyter
/Users/mikizhu/miniconda3/envs/py38_env/bin/jupyter notebook
```

