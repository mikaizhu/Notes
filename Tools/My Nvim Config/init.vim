set encoding=utf-8 fileencoding=utf-8 fileformats=unix,mac,dos
set fileencodings=utf-8,cp936,gb18030,big5,euc-jp,euc-kr,latin1
set mouse=a
" Appearance
set number norelativenumber background=dark display=lastline,uhex nowrap wrapmargin=0
set showmode shortmess+=I cmdheight=1 cmdwinheight=10 showbreak= breakindent breakindentopt=
set showmatch matchtime=0 matchpairs+=<:>,《:》,（:）,【:】,“:”,‘:’
set noshowcmd noruler rulerformat= laststatus=2
set title ruler titlelen=100 titleold= titlestring=%f noicon norightleft showtabline=2
set cursorline nocursorcolumn colorcolumn= concealcursor=nvc conceallevel=0
set list listchars=tab:\|\ ,extends:>,precedes:< synmaxcol=3000 ambiwidth=single
set nosplitbelow nosplitright nostartofline linespace=0 whichwrap=b,s scrolloff=5 sidescroll=0
set equalalways nowinfixwidth nowinfixheight winminwidth=3 winheight=3 winminheight=3
set termguicolors cpoptions+=I guioptions-=e nowarn noconfirm
set guicursor=n-v-c-sm:block,i-ci-ve:block,r-cr-o:hor20

" Clipboard
set clipboard=unnamed

" Performance
set updatetime=100 timeout timeoutlen=500 ttimeout ttimeoutlen=50 nolazyredraw

" neovim only
if matchstr(execute('silent version'), 'NVIM v\zs[^\n-]*') >= '0.4.0'
  set shada='20,<50,s10
  set inccommand=nosplit
  set wildoptions+=pum
  set signcolumn=yes:1
  set pumblend=0
endif
call plug#begin('~/.cache/nvim/plugged')
"nerd tree 
Plug 'preservim/nerdtree'
" Style
Plug 'Yggdroot/indentLine'
Plug 'kshenoy/vim-signature'
Plug 'lukas-reineke/indent-blankline.nvim' " can not exclude startify on the first :Startify
Plug 'guns/xterm-color-table.vim', {'on': 'XtermColorTable'}
Plug 'itchyny/lightline.vim'
Plug 'mengelbrecht/lightline-bufferline'
Plug 'mhinz/vim-startify', {'on': 'Startify'}
Plug 'ryanoasis/vim-devicons'
Plug 'itchyny/vim-cursorword'
" Git
Plug 'tpope/vim-fugitive'
Plug 'tpope/vim-git'
Plug 'iamcco/markdown-preview.nvim', {'for': 'markdown', 'do': 'cd app && npm install'}
call plug#end()
" put this after plugxxx, do not source colorscheme twice
colorscheme srcery
augroup FileTypeAutocmds
  autocmd!
  autocmd FileType * set formatoptions-=cro
  autocmd FileType * syntax sync minlines=50
  autocmd FileType *
        \ call matchadd('Special', '\W\zs\(TODO\|FIXME\|CHANGED\|XXX\|BUG\|HACK\)\ze') |
        \ call matchadd('Special', '\W\zs\(NOTE\|INFO\|IDEA\|NOTICE\|TMP\)\ze') |
        \ call matchadd('Special', '\W\zs\(DEBUG\|Debug\)\ze') |
        \ call matchadd('Special', '\W\zs\(@VOLDIKSS\|@voldikss\)\ze')
augroup END

augroup AutoSaveBuffer
  autocmd!
  " autocmd FocusLost,InsertLeave * call file#update()
  autocmd CursorHold * call file#update()
augroup END

