## vim官网

https://github.com/amix/vimrc

- 里面介绍了很多插件安装方法
- python vim https://github.com/Vimjas/vim-python-pep8-indent

**这个视频或许有帮助**

- https://www.youtube.com/watch?v=gnupOrSEikQ

## Nerd tree 使用

下面命令都要在目录窗口下使用

帮助菜单：？, 再按一次?关闭，操作和vim一样gg 和G等

关闭nerd tree：q

上下左右移动：hjkl

pane的切换：control + hjkl

打开目录：o，递归打开O，或者回车

折叠目录：x, 递归折叠 X

窗口的切换：

在新窗口中打开文件：t

上下切分pane：i

左右切分pane：s

官网教程：https://github.com/preservim/nerdtree/blob/master/doc/NERDTree.txt

## 更新vim

建议从官网更新最新版

- 下载对应版本

```
https://github.com/vim/vim/releases
```

这里建议使用下面操作下载：

```
sudo apt install vim
```

查看vim版本

```
vim --version
```

重定向命令

```
alias vi='/usr/local/bin/vim'
```

在`~/.bashrc`中最后一行添加命令

## 查看所有插件

在下面目录下查看插件

```
~/.vim/bundle
```

## How to install the Awesome version?

```
git clone --depth=1 https://github.com/amix/vimrc.git ~/.vim_runtime
sh ~/.vim_runtime/install_awesome_vimrc.sh
```

## 安装颜色

```
cd ~/.vim_runtime
# 插件更新
python update_plugins.py
```

## 文件目录插件NERDTree

首先安装插件管理软件vundle

- https://github.com/VundleVim/Vundle.vim
- https://github.com/VundleVim/Vundle.vim/blob/master/README_ZH_CN.md 中文文档

再安装插件

- https://github.com/preservim/nerdtree

参考教程：

- https://vimjc.com/vim-nerdtree-plugin.html

**如何快速使用？**

```
为了不每次都手动输入启动目录，可以在.vimrc中最后一行输入代码
au VimEnter * NERDTree
```

## 补全插件安装youcompleteme

如果安装不了这个插件：可以安装Vim插件jedi-vim，参考https://vimjc.com/jedi-vim-plugin.html

Jedi-vim下载， 这里建议手动安装：

```
cd ~/.vim
git clone http://github.com/davidhalter/jedi-vim path/to/bundles/jedi-vim
git clone --recursive http://github.com/davidhalter/jedi-vim
```

官网：https://github.com/ycm-core/YouCompleteMe#installation

安装：

- 更新了vim版本后，遇到下面问题，说明python版本不够，但检查了系统是正确的

```
requires Vim compiled with Python (3.6.0+) support
```

输入，发现python3是不支持的，可以发现前面是一个减号

```
vim --version
```

执行下面脚本：

```
#!/bin/bash

cd ~/vim
make clean
./configure --with-features=huge \
       --enable-python3interp=dynamic \
       --with-python3-config-dir=/usr/local/lib/python3.7/config-3.7m-x86_64-linux-gnu \
       --enable-cscope \
       --enable-gui=auto \
       --enable-gtk2-check \
       --enable-fontset \
       --enable-largefile \
       --disable-netbeans \
       --enable-fail-if-missing \
       --prefix=/usr/local \
       --with-x \

sudo make
sudo make install
```
执行完脚本后，提示下面错误：

```
YouCompleteMe：YouCompleteMe unavailable: unable to load Python
```

解决办法：

```
进入vim目录
make uninstall
make install
```

然后就可以运行了

参考：

- https://toutiao.io/posts/runvgs/preview
- https://blog.csdn.net/qq_42392366/article/details/107732973

然后又提示：

```
YouCompleteMe unavailable: No module named 'ycmd'
```

输入：

```
git submodule update --init --recursive
```

## python 风格

参考：

- https://vimjc.com/vim-python-ide.html

安装：

- todo

## Vim教程推荐

- https://zhuanlan.zhihu.com/p/360981919
- https://www.zhihu.com/question/444698010/answer/1733231285