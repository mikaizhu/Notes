# 说明：

现在使用neovim的比较多，而且里面的调用方式和vim差不多，只是函数的接口有变化，所以还是建议学习nvim

- vim的插件安装建议使用vim plug，不建议使用vundle
- 代码补全不建议使用ycm，建议用coc

## 学习文档：

- [coc vim：vim的补全，里面也有很多插件](https://github.com/neoclide/coc.nvim/wiki/Using-coc-extensions)
- [vim 学习文档](https://github.com/wsdjeg/vim-galore-zh_cn)
- [coc nvim](https://github.com/neoclide/coc.nvim)
- [vim视频教程](https://www.imooc.com/learn/1129)

# 教程


## vim的插件使用 

- 插件管理：vim plug 

第一个插件：vim-startify，可以打开最近打开的文件

- 寻找插件：1. 谷歌浏览器 2. vim awesome 3. 从别人的config文件中寻找

- 配置目录nerdtree

安装插件很简单,注意：该插件一般在文件的根目录下使用

按键映射：

```
"这里将tt按键设置为打开nerd tree
nnoremap tt :NERDTree<CR>
"因为要切换窗口，nerdtree默认使用control w，这里与tmux对应，c b h 左切换
nnoremap <C-b> <C-w>
```
在目录窗口，输入问号，可以打开操作说明

如果想要文件模糊搜索，可以尝试下ctrlp插件

- 配置轻量的python

使用插件python mode: https://github.com/python-mode/python-mode

注意详细使用请看官网

```
" python mode
let g:pymode_python = 'python3'
let g:pymode_trim_whitespaces = 1
let g:pymode_doc = 1
let g:pymode_doc_bind = 'K'
let g:pymode_rope_goto_definition_bind = "<C-]>"
let g:pymode_lint = 1
let g:pymode_lint_checkers = ['pyflakes', 'pep8', 'mccabe', 'pylint']
let g:pymode_options_max_line_length = 120
```

- 配置markdown

使用参考：https://sspai.com/post/60305
插件官网：https://github.com/iamcco/markdown-preview.nvim


