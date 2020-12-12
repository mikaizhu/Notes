# 基础设置

将vim设置成适合python编辑的编辑器

打开下面目录，vimrc文件是一个隐藏文件，可以对vim编辑器进行配置
```
vi ~/.vimrc
```

打开之后，输入下面配置
```
" => Chapter 1: Getting Started --------------------------------------- {{{

" Basic Python-friendly Vim configuration. Colorscheme is changed from
" 'default' to make screenshots look better in print.

syntax on                  " Enable syntax highlighting.
filetype plugin indent on  " Enable file type based indentation.

set autoindent             " Respect indentation when starting a new line.
set expandtab              " Expand tabs to spaces. Essential in Python.
set tabstop=4              " Number of spaces tab is counted for.
set shiftwidth=4           " Number of spaces to use for autoindent.

set backspace=2            " Fix backspace behavior on most terminals.

colorscheme murphy         " Change a colorscheme.
```
# 比较好的教程推荐

1. github中 https://github.com/PacktPublishing/Mastering-Vim
2. vim 指令大全 https://www.cnblogs.com/jeakon/archive/2012/08/13/2816802.html
