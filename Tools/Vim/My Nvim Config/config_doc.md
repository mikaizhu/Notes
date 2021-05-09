# 如果想关闭智能提示

```
if !exists('g:vscode') " TODO: use packer.nvim's `cond`
Plug 'neoclide/coc.nvim', {'branch': 'release'}
let g:coc_enabled = 1 # 0为开启智能提示
endif
```
# 如果想使用python3 vim

安装两个模块：使用conda 安装

- 搜索：conda pynvim
- 搜索：conda neovim 

配置环境变量：

let g:python3_host_prog='/Users/mikizhu/miniconda3/bin/python3.8'

如果出现了什么问题，可以通过:checkhelth进行检查

# vim markdown 设置

这里想设置markdown的分屏模式，即一半编辑，一半显示

指令安装：表示只有在markdown文件打开才有效
Plug 'instant-markdown/vim-instant-markdown', {'for': 'markdown'}

# vim 配置说明

如果要让vim配置生效，可以退出vim重启，或者重新source下
```
# 常见的默认操作
set nu
sntyx on
colorscheme hybird

# vim 映射
# leader键设置，因为键盘按键就这么多，使用leader键能够拓展按键。vim中映射比较
复杂，因为vim中有很多模式。
let mapleader = "," 常用的是逗号和空格

inoremap <leader>w <Esc>:w<CR> # inoremap 表示在插入模式下的映射，在插入模式下
只要输入 逗号 + w，然后就会触发后面的一系列的操作：回到普通模式，输入冒号加w，
回车。即保存操作。

# 基本映射map，使用map就可以实现按键映射,只在normal模式下有效

:map - x 输入冒号 map - x ，将减号映射为 x，
:map <space> viw 普通模式下按空格进入vision模式，并选中单词

# 取消映射
:unmap -  将刚刚映射的减号取消映射

# 模式映射
nmap/vmap/imap 只在normal/visul/insert模式下有效

提示：在vision模式下，使用U可以将选中的转换成大写

来看一个场景,这个时候如何工作呢
:nmap - dd
:nmap \ -
输入\后，就会启动dd命令，类似于递归操作

# 非递归映射
nnoremap/vnoremap/inoremap,任何时候都应该使用非递归映射
:nnoremap - dd
:nnoremap \ -
```

