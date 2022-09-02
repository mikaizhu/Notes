# 说明：

推荐教程：
- 笨方法学vimrc脚本。
- vim教程整理: https://github.com/vim-china/hello-vim

现在使用neovim的比较多，而且里面的调用方式和vim差不多，只是函数的接口有变化，所以还是建议学习nvim

- vim的插件安装建议使用vim plug，不建议使用vundle
- 代码补全不建议使用ycm，建议用coc

**常用命令记忆**：

gi：从普通模式，进入到最近的插入光标位置

control + ]:从insert 模式退回到普通模式，代替esc

control + w: hjkl 分屏模式下窗口切换

vim中有挺多介词，可以先尝试使用下面两个命令：

- ct)
- yt)
- ci)
- yi)
- di)
- dt)
- fi)
- ft)

单词跳转命令：

- e：跳转到单词结尾
- w：跳转到单词开头
- b:单词往前跳转
- W，E都是跳转，只不过是以空白为分隔

单词查找命令：

- f + char，比如在这一行使用f g 然后就会跳转到g开头的单词，使用,和；来查找上一
  个和下一个以g开头的单词

- t + char：跳转到这个字符的前一个字符位置, till 直到的意思

查找和删除命令结合：

this is vim:this is a test

假如要删除:前的单词，使用命令df:或者cf:但是注意光标一定要在:前面


删除和替换：

- c:change s:substitude r:replace

c命令和删除差不多，删除后进入编辑模式

测试：
“this is  a test”: cw,修改一个单词，但光标要在单词前面，如果光标在单词中间，则
用ciw，表示change in words

修改引号里面的单词

<change in words>:ci>,然后就可以修改括号里面的内容，并进入编辑模式

- C, S, R注意和小写的区别。C会删除后面所有字符，并进入到插入模式，S会删除整行
  ，并进入插入模式。R不会删除，只是会将后面的字符不断进行替换。

- 如果要删除引号里面的字符，或者括号里面的字符，可以尝试`df(`，

配置文件：

- noremap:按键映射，可以重新映射键盘按键

```
birenao n h "按n键相当于按h"
```

- map:命令与按键的映射
```
map S :w<CR> "将S映射为：w回车，即保存"
```

- set:表示设置属性，将某个属性开启或者关闭

```
set number "将行号进行开启"
```


vision 模式的作用：

- 可视行模式

- 可视块模式:control + v 进入可视块模式，假如现在要使用可视块模式在前面添加内
  容,选中你需要的行，然后I进入输入模式，输入test文字后，被可视化选中的段落前面
  ，都会添加上字符。

test:this is test:hello
test:this is test:hello
test:this is test:hello

- 可视的normal模式:假如我要在下面四行前面都加上某些字符，使用方式为按冒号：+
  normal即可进入可视的普通模式，然后输入你要的指令，完整指令如下：`:normal
  Ithis is test:`, I表示在最前面插入，后面为插入的字符

this is test:hello 
this is test:hello
this is test:hello
this is test:hello

# 学习文档：

- [coc vim：vim的补全，里面也有很多插件](https://github.com/neoclide/coc.nvim/wiki/Using-coc-extensions)
- [vim 学习文档](https://github.com/wsdjeg/vim-galore-zh_cn)
- [coc nvim](https://github.com/neoclide/coc.nvim)
- [vim视频教程](https://www.imooc.com/learn/1129)

# 教程

vim && markdown 教程

插件安装：
- markdown previe

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

nerdtree 中的文件控制：可以直接在nerdtree中新建文件，在目录中使用ma命令，创建
文件。参考：https://blog.csdn.net/qq_38883889/article/details/107014964


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

# vim标签页

即vim可以同时打开很多个文件， 即 Tab page

如果是直接使用vi file1 file2 这样打开多个文件，那么只能使用下面方式切换

```
:bn
:bp
:b2
```

1. 建立新的标签页或者打开新的文件

```
:tabnew filename
:tabe filename
:tabedit filename
```

2. 切换标签页

参考：https://vimjc.com/vim-tabpage.html

在普通模式下：

```
gt
gT
```

在命令模式下：

```
:tabnext
:tabn
:tabprevious
:tabp
```

3. 关闭标签页

```
:tabclose
# 下面命令只保留当前的标签页，关闭其他标签页
:tabo
:tabonly
```
# vim替换命令

全局替换：
```
:%s/old/new/g
:%s/old/new
```

几行替换：
```
# 1-5行进行替换
:1,5s/old/new
```

指定几行替换：

```

```

# 删除命令

```
:1,5d
```

