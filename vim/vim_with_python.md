# 基础设置

将vim设置成适合python编辑的编辑器

打开下面目录，vimrc文件是一个隐藏文件，可以对vim编辑器进行配置
```
vi ~/.vimrc
```

打开之后，输入下面配置
```
" => Chapter 1: Getting Started --------------------------------------- {{{

syntax on                  " Enable syntax highlighting.
filetype plugin indent on  " Enable file type based indentation.

set autoindent             " Respect indentation when starting a new line.
set expandtab              " Expand tabs to spaces. Essential in Python.
set tabstop=4              " Number of spaces tab is counted for.
set shiftwidth=4           " Number of spaces to use for autoindent.

set backspace=2            " Fix backspace behavior on most terminals.

colorscheme murphy         " Change a colorscheme.

" => Chapter 2: Advanced Movement and Navigation ---------------------- {{{

packloadall                     " Load all plugins.
silent! helptags ALL            " Load help files for all plugins.

" Navigate windows with <Ctrl-hjkl> instead of <Ctrl-w> followed by hjkl.
noremap <c-h> <c-w><c-h>
noremap <c-j> <c-w><c-j>
noremap <c-k> <c-w><c-k>
noremap <c-l> <c-w><c-l>

set foldmethod=indent           " Indentation-based folding.

set wildmenu                    " Enable enhanced tab autocomplete.
set wildmode=list:longest,full  " Complete till longest string, then open menu.

" Plugin-related settings below are commented out. Uncomment them to enable
" the plugin functionality once you download the plugins.

" let NERDTreeShowBookmarks = 1   " Display bookmarks on startup.
" autocmd VimEnter * NERDTree     " Enable NERDTree on Vim startup.
" Autoclose NERDTree if it's the only open window left.
" autocmd bufenter * if (winnr("$") == 1 && exists("b:NERDTree") &&
"   \ b:NERDTree.isTabTree()) | q | endif

" set number                      " Display column numbers.
" set relativenumber              " Display relative column numbers.

set hlsearch                    " Highlight search results.
set incsearch                   " Search as you type.

set clipboard=unnamed,unnamedplus  " Copy into system (*, +) registers.
```
# 比较好的教程推荐

1. github中 https://github.com/PacktPublishing/Mastering-Vim
2. vim 指令大全 https://www.cnblogs.com/jeakon/archive/2012/08/13/2816802.html
