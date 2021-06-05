ubuntu 下 .bashrc文件的作用

要想用户每次登陆都会自动执行某些命令，则可以修改`vi ~/.bash_profile`，这个文件只对当前用户有效。

- todo

要让.bashrc每次都自动运行，要修改

```/
vi ~/.bash_profile

# 添加下面代码
if test -f .bashrc ; then
source .bashrc
fi
```

然后

```
source ~/.bash_profile
```

接下来打开窗口还是tmux新窗口，都会执行bashrc文件

里面配置如下：

```
if test -f ~/.local/bin/bashmarks.sh; then
    source ~/.local/bin/bashmarks.sh
fi

if test -f .bashrc ; then                                                                                                   source .bashrc                                                                                                     fi
```

