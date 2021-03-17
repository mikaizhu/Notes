# vim插件安装

主要是将vim打造成可以编写python环境的软件

- 参考教程1 https://www.jianshu.com/p/f0513d18742a

主要介绍了如何配置vim，让vim可以编写python。然后卡在补全插件的安装

- 参考教程2 https://www.jianshu.com/p/d908ce81017a

介绍了补全插件的安装，介绍了两种方法，主要通过git安装比较靠谱

# 安装步骤

不管是Mac系统还是ubuntu系统

目前步骤都是一样的

1. 下载vim的插件管理器Vundle

```
https://github.com/VundleVim/Vundle.vim
```

可以查看github的下面说明

注意需要clone到下面地址，没有就自己创建

```
~/.vim/bundle/Vundle.vim
```

2. 然后在home目录下创建.vimrc文件。

3. 将下面内容复制进去

```
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'
" The sparkup vim script is in a subdirectory of this repo called vim.
" Pass the path to set the runtimepath properly.
Plugin 'rstacruz/sparkup', {'rtp': 'vim/'}
" Install L9 and avoid a Naming conflict if you've already installed a
" different version somewhere else.
" Plugin 'ascenator/L9', {'name': 'newL9'}

" All of your Plugins must be added before the following line
call vundle#end()            " required
filetype plugin indent on    " required
" To ignore plugin indent changes, instead use:
"filetype plugin on
"
" Brief help
" :PluginList       - lists configured plugins
" :PluginInstall    - installs plugins; append `!` to update or just :PluginUpdate
" :PluginSearch foo - searches for foo; append `!` to refresh local cache
" :PluginClean      - confirms removal of unused plugins; append `!` to auto-approve removal
"
" see :h vundle for more details or wiki for FAQ
" Put your non-Plugin stuff after this line
```

4. 先关闭下vim，然后用vim随便打开一个文件。输入下面命令

```
:PluginInstall
```

5. 然后安装插件YouCompleteMe

# 插件的安装

先进入插件的官网，查看下载步骤

```
https://github.com/ycm-core/YouCompleteMe
```

## Mac插件安装

```
https://github.com/ycm-core/YouCompleteMe#macos
```

## Linux插件安装

```
https://github.com/ycm-core/YouCompleteMe#linux-64-bit
```