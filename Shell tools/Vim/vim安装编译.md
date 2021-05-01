如果之前下载了vim，那么可以先删除：

```
cd /vim/src
sudo make uninstall
```

从官网下载对应的版本

- https://github.com/vim/vim/releases

```
unzip **.zip && move vim**_file vim
cd vim
cd src
make clean
```

参考：

- https://toutiao.io/posts/runvgs/preview
- https://rovo98.github.io/posts/97c4fd12/

找到python的config文件位置

```
/usr/local/bin/python3-config --configdir
```

**重置vim编译**，如果觉得没有编译好，可以重置：

```
# 进入到vim的目录下
cd ~/vim
sudo make distclean
```

**进行config配置**

```
#!/bin/bash
# 先安装必要插件
sudo apt install
    libcairo2-dev libx11-dev libxpm-dev libxt-dev  \
            python3-dev ruby-dev  libperl-dev git

# 使用以下配置即可，下面是支持python3的
./configure --with-features=huge \
        --enable-multibyte \
        --enable-rubyinterp=yes \
        --enable-python3interp=yes \
        --enable-perlinterp=yes \
```

**开始编译**

```
sudo make
sudo make install
```

**查看是否支持python**

```
vim --version
```

**如果要支持复制粘贴板，使用下面命令**

先安装下面模块

```
sudo apt install libx11-dev libxtst-dev libxt-dev libsm-dev libxpm-dev
```

然后开始编译

```
./configure --with-features=huge \
        --enable-multibyte \
        --enable-rubyinterp=yes \
        --enable-python3interp=yes \
        --enable-perlinterp=yes \
        --with-x
sudo make
sudo make install
```

但是ssh的复制粘贴板还是不能使用，比如我在vim中使用y复制，然后在Mac上想继续复制粘贴。当vim支持clipborad，复制粘贴只能继续在服务器上使用。

