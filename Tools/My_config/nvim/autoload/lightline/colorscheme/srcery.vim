" -----------------------------------------------------------------------------
" File: srcery.vim
" Description: Srcery colorscheme for Lightline (itchyny/lightline.vim)
" Author: Roosta <mail@roosta>
" Source: https://github.com/morhetz/gruvbox/blob/master/autoload/lightline/colorscheme/gruvbox.vim
" Last Modified: 2017-03-28
" -----------------------------------------------------------------------------

function! s:getSrcColor(group) abort
  let l:guiColor = synIDattr(hlID(a:group), 'fg', 'gui')
  let l:termColor = synIDattr(hlID(a:group), 'fg', 'cterm')
  return [ l:guiColor, l:termColor ]
endfunc

if g:colors_name == 'srcery'
  let s:black        = s:getSrcColor('SrceryBlack')
  let s:bright_white = s:getSrcColor('SrceryBrightWhite')
  let s:bright_black = s:getSrcColor('SrceryBrightBlack')
  let s:white        = s:getSrcColor('SrceryWhite')
  let s:yellow       = s:getSrcColor('SrceryYellow')
  let s:blue         = s:getSrcColor('SrceryBlue')
  let s:bright_blue  = s:getSrcColor('SrceryBrightBlue')
  let s:cyan         = s:getSrcColor('SrceryCyan')
  let s:bright_cyan  = s:getSrcColor('SrceryBrightCyan')
  let s:red          = s:getSrcColor('SrceryRed')
  let s:bright_red   = s:getSrcColor('SrceryBrightRed')
  let s:orange       = s:getSrcColor('SrceryOrange')
  let s:magenta      = s:getSrcColor('SrceryMagenta')
  let s:green        = s:getSrcColor('SrceryGreen')
  let s:bright_green = s:getSrcColor('SrceryBrightGreen')
  let s:xgray1       = s:getSrcColor('SrceryXgray1')
  let s:xgray2       = s:getSrcColor('SrceryXgray2')
  let s:xgray3       = s:getSrcColor('SrceryXgray3')
  let s:xgray4       = s:getSrcColor('SrceryXgray4')
  let s:xgray5       = s:getSrcColor('SrceryXgray5')
else
  let s:black        = ['#1C1B19', '0']
  let s:bright_white = ['#FCE8C3', '15']
  let s:bright_black = ['#2D2C29', '8']
  let s:white        = ['#918175', '7']
  let s:yellow       = ['#FBB829', '3']
  let s:blue         = ['#2C78BF', '4']
  let s:bright_blue  = ['#68A8E4', '12']
  let s:cyan         = ['#0AAEB3', '6']
  let s:bright_cyan  = ['#53FDE9', '14']
  let s:red          = ['#EF2F27', '1']
  let s:bright_red   = ['#F75341', '9']
  let s:orange       = ['#FF5F00', '202']
  let s:magenta      = ['#E02C6D', '5']
  let s:green        = ['#519F50', '2']
  let s:bright_green = ['#98BC37', '10']
  let s:xgray1       = ['#262626', '235']
  let s:xgray2       = ['#303030', '236']
  let s:xgray3       = ['#3A3A3A', '237']
  let s:xgray4       = ['#444444', '238']
  let s:xgray5       = ['#4E4E4E', '239']
endif

let s:p = {'normal':{}, 'inactive':{}, 'insert':{}, 'replace':{}, 'visual':{}, 'tabline':{}, 'terminal':{}, 'command':{}}
let s:p.normal.left     = [ [ s:bright_white, s:xgray4 ], [ s:bright_white, s:xgray3 ] ]
let s:p.normal.right    = [ [ s:bright_white, s:xgray4 ], [ s:bright_white, s:xgray3 ] ]
let s:p.normal.middle   = [ [ s:bright_white, s:bright_black ] ]
let s:p.inactive.right  = [ [ s:white, s:bright_black ], [ s:white, s:bright_black ] ]
let s:p.inactive.left   = [ [ s:white, s:bright_black ], [ s:white, s:bright_black ] ]
let s:p.inactive.middle = [ [ s:xgray5, s:bright_black ] ]
let s:p.insert.left     = [ [ s:black, s:bright_white ], [ s:black, s:white ] ]
let s:p.insert.right    = [ [ s:black, s:bright_white ], [ s:black, s:white ] ]
let s:p.insert.middle   = [ [ s:bright_white, s:bright_black ] ]
let s:p.replace.left    = [ [ s:bright_white, s:bright_red ], [ s:black, s:white ] ]
let s:p.replace.right   = [ [ s:bright_white, s:bright_red ], [ s:black, s:white ] ]
let s:p.replace.middle  = [ [ s:bright_white, s:bright_black ] ]
let s:p.visual.left     = [ [ s:black, s:cyan ], [ s:bright_white, s:xgray4 ] ]
let s:p.visual.right    = [ [ s:black, s:cyan ], [ s:bright_white, s:xgray4 ] ]
let s:p.visual.middle   = [ [ s:bright_white, s:bright_black ] ]
let s:p.tabline.left    = [ [ s:black, s:white ] ]
let s:p.tabline.tabsel  = [ [ s:bright_blue, s:bright_black ] ]
let s:p.tabline.middle  = [ [ s:black, s:bright_black ] ]
let s:p.tabline.right   = [ [ s:black, s:bright_white ] ]
let s:p.normal.error    = [ [ s:bright_white, s:red ] ]
let s:p.normal.warning  = [ [ s:black, s:orange ] ]
let s:p.terminal.left   = [ [ s:black, s:green ], [ s:bright_white, s:xgray4 ] ]
let s:p.terminal.right  = [ [ s:black, s:green ], [ s:bright_white, s:xgray4 ] ]
let s:p.terminal.middle = [ [ s:bright_white, s:bright_black ] ]
let s:p.command.left    = [ [ s:black, s:yellow ], [ s:bright_white, s:xgray4 ] ]
let s:p.command.right   = [ [ s:black, s:yellow ], [ s:bright_white, s:xgray4 ] ]
let s:p.command.middle  = [ [ s:bright_white, s:bright_black ] ]


let g:lightline#colorscheme#srcery#palette = lightline#colorscheme#flatten(s:p)
