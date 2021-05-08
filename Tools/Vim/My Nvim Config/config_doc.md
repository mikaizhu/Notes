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
