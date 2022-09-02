" vim:et:ts=4:sw=4:fdm=marker:fmr={{{,}}}
"=============================================================================
" File: color.vim
" Author: romgrk
" Date: 15 Mar 2016
" Description: vimscript RGB/HSL color parsing
" This is mostly a transcript from a javascript color parsing module.
" !::exe [so %]
" ================================================================================
" Definitions: {{{1
" In function names:
"  • “Hex” refers to hexadecimal color format e.g. #599eff
"  • “RGB” refers to an array of numbers [r, g, b]
"                                      where   r,g,b ∈ [0, 255]
"  • “HSL” refers to an array of floats  [h, s, l]
"                                      where   h,s,l ∈ [0, 1.0]
" }}}1
" ================================================================================
" Color-format patterns:
" {{{1

let s:patterns = {}

" 6 hex-numbers, optionnal #-prefix
let s:patterns['hex']      = '\v#?(\x{2})(\x{2})(\x{2})'

" short version is strict: starting # mandatory
let s:patterns['shortHex'] = '\v#(\x{1})(\x{1})(\x{1})'

" Disabled
"let s:patterns['rgb']  = '\vrgb\s*\((\d+)\s*,(\d+)\s*,(\d+)\s*)\s*'
"let s:patterns['rgba'] = '\vrgba\s*\((\d+)\s*,(\d+)\s*,(\d+)\s*,(\d+)\s*)\s*'

" }}}1
" ================================================================================
" Functions:
" {{{1

" @params  String       string   The string to test
" @returns Boolean     [0 or 1]  if string matches: rrggbb OR #rrggbb OR #rgb
fu! s:color__Test (string)
    for key in keys(s:patterns)
        if a:string =~# s:patterns[key]
            return 1
        end
    endfor
    return 0
endfu

" @params (r, g, b)
" @params ([r, g, b])
" @returns String           A RGB color
fu! s:color__RGBtoHex (...)
    let [r, g, b] = ( a:0==1 ? a:1 : a:000 )
    let num = printf('%02x', float2nr(r)) . ''
          \ . printf('%02x', float2nr(g)) . ''
          \ . printf('%02x', float2nr(b)) . ''
    return '#' . num
endfu

" @param {String|Number} color   The color to parse
fu! s:color__HexToRGB (color)
    if type(a:color) == 2
        let color = printf('%x', a:color)
    else
        let color = a:color | end

    let matches = matchlist(color, s:patterns['hex'])
    let factor  = 0x1

    if empty(matches)
        let matches = matchlist(color, s:patterns['shortHex'])
        let factor  = 0x10
    end

    if len(matches) < 4
        throw 'Couldnt parse ' . string(color) . ' ' .  string(matches)
    end

    let r = str2nr(matches[1], 16) * factor
    let g = str2nr(matches[2], 16) * factor
    let b = str2nr(matches[3], 16) * factor

    return [r, g, b]
endfu


" Converts an HSL color value to RGB. Conversion formula
" adapted from http://en.wikipedia.org/wiki/HSL_color_space.
" Assumes h, s, and l are contained in the set [0, 1] and
" returns r, g, and b in the set [0, 255].
" @param   Number  h     OR     @param  Array [h, s, l]
" @param   Number  s
" @param   Number  l
" @returns Array [r, g, b]     The RGB representation
fu! s:color__HSLtoRGB(...) " (h, s, l)
    let [h, s, l] = ( a:0==1 ? a:1 : a:000 )

    if (s == 0.0) " achromatic
        let r = l
        let g = l
        let b = l
    else
        let q = l < 0.5 ? l * (1 + s) : l + s - l * s
        let p = 2 * l - q
        let r = s:color__Hue2RGB(p, q, h + 0.33333)
        let g = s:color__Hue2RGB(p, q, h)
        let b = s:color__Hue2RGB(p, q, h - 0.33333)
    end

    return [r * 255.0, g * 255.0, b * 255.0]
endfu


" @param   Number  r     OR     @param  Array [r, g, b]
" @param   Number  g
" @param   Number  b
" @returns Array [h, s, l]     The HSL representation
fu! s:color__RGBtoHSL(...)
    let [r, g, b] = ( a:0==1 ? a:1 : a:000 )
    let max = max([r, g, b])
    let min = min([r, g, b])

    let r   = str2float(r)
    let g   = str2float(g)
    let b   = str2float(b)
    let max = str2float(max)
    let min = str2float(min)

    let max = max / 255
    let min = min / 255
    let r = r / 255
    let g = g / 255
    let b = b / 255
    let h = str2float(0)
    let s = str2float(0)
    let l = (max + min) / 2

    if (max == min)
        let h = 0   " achromatic
        let s = 0   " achromatic
    else
        let d = max - min
        let s = (l > 0.5 ? d / (2 - max - min)
                       \ : d / (max + min)     )
        if (max == r)
            let h = (g - b) / d + (g < b ? 6 : 0)
        end
        if (max == g)
            let h = (b - r) / d + 2
        end
        if (max == b)
            let h = (r - g) / d + 4
        end
        let h = h / 6
    end

    return [h, s, l]
endfu

fu! s:color__Hue2RGB(...) " (p, q, t)
    let [p, q, t] = ( a:0==1 ? a:1 : a:000 )

    if(t < 0) | let t += 1 | end
    if(t > 1) | let t -= 1 | end

    if(t < 1.0/6) | return (p + (q - p) * 6.0 * t)           | end
    if(t < 1.0/2) | return (q)                               | end
    if(t < 2.0/3) | return (p + (q - p) * (2.0/3 - t) * 6.0) | end

    return p
endfu

" }}}1
" ================================================================================
" Composed functions:
" {{{1

fu! s:color__HexToHSL (color)
    let [r, g, b] = s:color__HexToRGB(a:color)
    return s:color__RGBtoHSL(r, g, b)
endfu

fu! s:color__HSLtoHex (...)
    let [h, s, l] = ( a:0==1 ? a:1 : a:000 )
    let [r, g, b] = s:color__HSLtoRGB(h, s, l)
    return s:color__RGBtoHex(r, g, b)
endfu

" @params String                 color      The color
" @params {Number|String|Float} [amount=5]  The percentage of light
fu! s:color__Lighten(color, ...)
    let amount = a:0 ?
                \(type(a:1) < 2 ?
                    \str2float(a:1) : a:1 )
                \: 5.0

    if(amount < 1.0)
        let amount = 1.0 + amount
    else
        let amount = 1.0 + (amount / 100.0)
    end

    let rgb = s:color__HexToRGB(a:color)
    let rgb = map(rgb, 'v:val * amount')
    let rgb = map(rgb, 'v:val > 255.0 ? 255.0 : v:val')
    let rgb = map(rgb, 'float2nr(v:val)')
    let hex = s:color__RGBtoHex(rgb)
    return hex
endfu

" @params String                 color      The color
" @params {Number|String|Float} [amount=5]  The percentage of darkness
fu! s:color__Darken(color, ...)
    let amount = a:0 ?
                \(type(a:1) < 2 ?
                    \str2float(a:1) : a:1 )
                \: 5.0

    if(amount < 1.0)
        let amount = 1.0 - amount
    else
        let amount = 1.0 - (amount / 100.0)
    end

    if(amount < 0.0)
        let amount = 0.0 | end


    let rgb = s:color__HexToRGB(a:color)
    let rgb = map(rgb, 'v:val * amount')
    let rgb = map(rgb, 'v:val > 255.0 ? 255.0 : v:val')
    let rgb = map(rgb, 'float2nr(v:val)')
    let hex = s:color__RGBtoHex(rgb)
    return hex
endfu

function! s:color__Decrease (...)
    if &background == 'light'
        return call(function('s:color__Lighten'), a:000)
    end
    return call(function('s:color__Darken'), a:000)
endfunc

function! s:color__Increase (...)
    if &background == 'light'
        return call(function('s:color__Darken'), a:000)
    end
    return call(function('s:color__Lighten'), a:000)
endfunc

function! s:color__Mix (a, b, ...)
    let amount = a:0 ? a:1 : 0.5

    let ca = s:color__HexToRGB(a:a)
    let cb = s:color__HexToRGB(a:b)

    let r = s:interpolate(ca[0], cb[0], amount)
    let g = s:interpolate(ca[1], cb[1], amount)
    let b = s:interpolate(ca[2], cb[2], amount)

    return s:color__RGBtoHex([r, g, b])
endfunc

function! s:interpolate (start, end, amount)
    let diff = a:end - a:start
    return a:start + (diff * a:amount)
endfunc

" }}}1


"=============================================================================
" File: doom-one.vim
" Author: romgrk
"=============================================================================
" !::exe [So]

if &background != 'dark'
  set background=dark
end

let colors_name = 'doom-one'

" Highlighting function {{{
function! s:_ (name, ...)
  let fg = ''
  let bg = ''
  let attr = ''

  if type(a:1) == 3
    let fg   = get(a:1, 0, '')
    let bg   = get(a:1, 1, '')
    let attr = get(a:1, 2, '')
  else
    let fg   = get(a:000, 0, '')
    let bg   = get(a:000, 1, '')
    let attr = get(a:000, 2, '')
  end

  let has_props = v:false

  let cmd = 'hi! ' . a:name
  if !empty(fg) && fg != 'none'
    let cmd .= ' guifg=' . fg
    let has_props = v:true
  end
  if !empty(bg) && bg != 'none'
    let cmd .= ' guibg=' . bg
    let has_props = v:true
  end
  if !empty(attr) && attr != 'none'
    let cmd .= ' gui=' . attr
    let has_props = v:true
  end
  execute 'hi! clear ' a:name
  if has_props
    execute cmd
  end
endfunc

" }}}
" Colors                                                                     {{{

let s:base0      = '#1B2229'
let s:base1      = '#1c1f24'
let s:base2      = '#202328'
let s:base3      = '#23272e'
let s:base4      = '#3f444a'
let s:base5      = '#5B6268'
let s:base6      = '#73797e'
let s:base7      = '#9ca0a4'
let s:base8      = '#b1b1b1'
let s:base9      = '#E6E6E6'

let s:grey       = s:base4
let s:red        = '#ff6c6b'
let s:orange     = '#da8548'
let s:green      = '#98be65'
let s:teal       = '#4db5bd'
let s:yellow     = '#ECBE7B'
let s:blue       = '#51afef'
let s:dark_blue  = '#2257A0'
let s:magenta    = '#c678dd'
let s:violet     = '#a9a1e1'
let s:cyan       = '#46D9FF'
let s:dark_cyan  = '#5699AF'
let s:white      = '#efefef'

let s:green_alt  = '#799033'

let s:bg             = '#282c34'
let s:bg_alt         = '#21242b'
let s:bg_highlight   = '#2E323C'
let s:bg_popup       = '#3E4556'
let s:bg_widget      = s:bg
let s:bg_statusline  = s:bg_popup
let s:bg_visual      = s:color__Lighten(s:base4, 0.3)
let s:bg_selection   = s:dark_blue
let s:bg_highlighted = '#4A4A45'

let s:fg           = '#bbc2cf'
let s:fg_alt       = '#5B6268'
let s:fg_widget    = s:fg
let s:fg_conceal   = s:base4
let s:fg_subtle    = s:base7
let s:fg_highlight = s:color__Lighten(s:fg, 0.2)
let s:fg_linenr    = s:base4


let s:highlight       = s:blue
let s:highlight_color = s:base0

let s:tag    = s:color__Mix(s:blue, s:cyan)

let s:diff_info_fg  = s:blue
let s:diff_info_bg0 = s:color__Mix('#D8EEFD', s:bg, 0.6)
let s:diff_info_bg1 = s:color__Mix('#D8EEFD', s:bg, 0.8)

let s:diff_add_fg  = s:green
let s:diff_add_fg0 = s:color__Mix(s:green,   s:fg, 0.4)
let s:diff_add_bg0 = s:color__Mix('#506d5b', s:bg, 0.4)
let s:diff_add_bg1 = s:color__Mix('#acf2bd', s:bg, 0.6)
let s:diff_add_bg2 = s:color__Mix('#acf2bd', s:bg, 0.8)

let s:gh_danger_fg  = s:red
let s:gh_danger_fg0 = s:color__Mix(s:red,     s:fg, 0.6)
let s:gh_danger_bg0 = s:color__Mix('#ffdce0', s:bg, 0.6)
let s:gh_danger_bg1 = s:color__Mix('#ffdce0', s:bg, 0.8)
let s:gh_danger_bg2 = s:color__Mix('#ffdce0', s:bg, 0.9)

" }}}
" General UI                                                                 {{{

call s:_('Normal',            s:fg,           s:bg)
call s:_('EndOfBuffer',       s:bg,           s:bg)
call s:_('NormalPopup',       s:fg_highlight, s:bg_popup)
call s:_('NormalPopover',     s:fg_highlight, s:bg_popup)
call s:_('NormalPopupPrompt', s:base7,        s:color__Darken(s:bg_popup, 0.3), 'bold')
call s:_('NormalPopupSubtle', s:base6,        s:bg_popup)

call s:_('Cursor',           '', s:base0,  'reverse')
call s:_('SecondaryCursor',  '', s:highlight, 'none')

call s:_('Folded',           s:base7,  s:bg_highlight, 'none')
call s:_('FoldColumn',       s:fg_alt, s:bg_widget, '')
call s:_('SignColumn',       '',       s:bg_widget, '')
call s:_('ColorColumn',      '',       s:bg_highlight, '')

call s:_('CursorLine',       '',          s:bg_highlight)
call s:_('CursorColumn',     '',          s:bg_highlight)
call s:_('CursorLineNr',     s:highlight, s:bg_highlight, 'none')
call s:_('LineNr',           s:fg_linenr, s:bg_widget,    'none')

call s:_('IndentGuide',      s:base4, '', '')
call s:_('IndentGuidesEven', s:base4, '', '')
call s:_('IndentGuidesOdd',  s:base4, '', '')

call s:_('TermCursor',       s:fg,        'none',            'reverse')
call s:_('TermCursorNC',     s:fg_alt,    'none',            'reverse')
call s:_('TermNormal',       s:fg,        s:bg,    '')
hi! link TermNormalNC TermNormal


call s:_('Visual',         '', s:bg_visual)
call s:_('VisualBold',     '', s:bg_visual, 'bold')

call s:_('WildMenu',       s:fg,            s:bg_selection)

call s:_('StatusLine',       s:base8,  s:bg_statusline, 'none')
call s:_('StatusLineNC',     s:base6,  s:bg_statusline, 'none')
call s:_('StatusLinePart',   s:base6,  s:bg_statusline, 'bold')
call s:_('StatusLinePartNC', s:base6,  s:bg_statusline, 'bold')

call s:_('Separator', s:fg_alt, 'none')
call s:_('VertSplit', s:base4,   s:bg, 'none')

call s:_('Pmenu',        s:fg,              s:bg_popup)
call s:_('PmenuSel',     s:highlight_color, s:highlight)
call s:_('PmenuSelBold', s:highlight_color, s:highlight, 'bold')
call s:_('PmenuSbar',    '',                s:bg_alt)
call s:_('PmenuThumb',   '#666660',         '#666660')

if exists('&pumblend')
  set pumblend=0
end

let s:bg_current = s:bg
let s:bg_visible = s:bg
let s:bg_other   = s:base1

call s:_('TabLine',             s:base7, s:bg_alt,  'bold')
call s:_('TabLineSel',          s:blue,  s:bg_current, 'bold')
call s:_('TabLineFill',         'none',  s:bg_other,   'bold')

call s:_('BufferCurrent',       s:base9,          s:bg_current,  'none')
call s:_('BufferCurrentIndex',  s:base6,          s:bg_current,  'none')
call s:_('BufferCurrentMod',    s:yellow,         s:bg_current,  'none')
call s:_('BufferCurrentSign',   s:blue,           s:bg_current,  'none')
call s:_('BufferCurrentTarget', s:red,            s:bg_current,  'bold')

call s:_('BufferVisible',       s:base7,          s:bg_visible,  'none')
call s:_('BufferVisibleIndex',  s:base9,          s:bg_visible,  'none')
call s:_('BufferVisibleMod',    s:yellow,         s:bg_visible,  'none')
call s:_('BufferVisibleSign',   s:base4,          s:bg_visible,  'none')
call s:_('BufferVisibleTarget', s:red,            s:bg_visible,  'bold')

call s:_('BufferInactive',       s:base6,          s:bg_other,    'none')
call s:_('BufferInactiveIndex',  s:base6,          s:bg_other,    'none')
call s:_('BufferInactiveMod',    s:yellow,         s:bg_other,    'none')
call s:_('BufferInactiveSign',   s:base4,          s:bg_other,    'none')
call s:_('BufferInactiveTarget', s:red,            s:bg_other,    'bold')

call s:_('BufferTabpages',       s:blue,           s:bg_statusline, 'bold')
call s:_('BufferTabpageFill',    s:base4,          s:bg_other,    'bold')

call s:_('BufferPart',        s:diff_info_fg,   s:diff_info_bg0, 'bold')

" }}}
" Search, Highlight, Conceal, Messages                                       {{{

" let s:bg_search = s:color__Mix(s:bg, s:yellow, 0.2)
let s:bg_search = s:yellow

call s:_('Search',          s:highlight_color, s:bg_search,      'none')
call s:_('IncSearch',       s:highlight_color, s:bg_search,      'none')
call s:_('IncSearchCursor', '', '',               'reverse')

call s:_('Conceal',         s:fg_conceal, 'none', '')
call s:_('SpecialKey',      s:violet,     'none', 'bold')
call s:_('NonText',         s:fg_alt,     '',     'bold')
call s:_('MatchParen',      s:red,        'none', 'bold')
call s:_('Whitespace',      s:base4,      '',     '')


call s:_('Highlight',       '', s:bg_highlighted, 'none')
call s:_('HighlightSubtle', '', s:bg_highlighted, 'none')

call s:_('Question',        s:green, '', 'bold')

call s:_('File',            s:fg,      '', 'bold')
call s:_('Directory',       s:yellow,  '', 'bold')
call s:_('Section',         s:magenta, '', 'bold')
call s:_('Title',           s:violet,  '', 'bold')

call s:_('Bold', '', '', 'bold')

" Text levels                                                                {{{

let s:text_colors = {
\ 'Normal':  s:fg,
\ 'Info':    s:blue,
\ 'Success': s:green,
\ 'Warning': s:yellow,
\ 'Debug':   s:yellow,
\ 'Error':   s:red,
\ 'Special': s:violet,
\ 'Muted':   s:base7,
\}
for key in keys(s:text_colors)
  call s:_('Text' . key,          s:text_colors[key], '', '')
  call s:_('Text' . key . 'Bold', s:text_colors[key], '', 'bold')
endfor

" NOTE/DONE/TODO/FIXME/XXX/DEPRECATED comment highlights

" }}}

hi! link Msg        TextSuccess
hi! link MoreMsg    TextInfo
hi! link WarningMsg TextWarning
hi! link ErrorMsg   TextError
hi! link Error      TextError
hi! link ModeMsg    TextSpecial
hi! link Todo       TextWarningBold


" }}}
" Main Syntax                                                               {{{1

call s:_('Tag',                  s:tag, '',        'underline')
call s:_('Link',                 s:tag, '',        'underline')
call s:_('URL',                  s:tag, '',        'underline')
hi! link Underlined     URL

call s:_('Comment',              s:base6, '',        '')
call s:_('CommentBold',          s:base6, '',        'bold')
call s:_('SpecialComment',       s:base7, '',        'bold')
hi! link CommentURL     URL
hi! link CommentLabel   CommentBold
hi! link CommentSection CommentBold
hi! link Noise Comment

call s:_('Global',               s:violet, '',         'bold')
call s:_('PreProc',              s:magenta, '',        'none')
call s:_('Macro',                s:magenta, '',        'bold')
call s:_('Define',               s:magenta, '',        'bold')
call s:_('PreCondit',            s:magenta, '',        'bold')
call s:_('Include',              s:magenta, '',        'bold')

call s:_('Repeat',               s:blue, '',        '')
call s:_('Keyword',              s:blue, '',        '')
call s:_('Statement',            s:blue, '',        'none')
call s:_('Label',                s:blue, '',        '')

call s:_('Operator',             s:blue, '',        '')

call s:_('Constant',             s:violet, '',        'bold')

call s:_('Number',               s:orange, '',        'none')
call s:_('Float',                s:orange, '',        'none')
call s:_('Boolean',              s:orange, '',        'none')
call s:_('Enum',                 s:orange, '',        'none')

call s:_('Delimiter',            s:blue,  '',        'none')
call s:_('DelimiterAlt',         s:green_alt, '',    'none')
call s:_('SpecialChar',          s:base8, '',        'bold')

call s:_('String',               s:green, '',        'none')
call s:_('StringDelimiter',      s:green, '',        'none')

call s:_('Character',            s:green, '',        'bold')


call s:_('Special',              s:violet, '',        'bold')
call s:_('SpecialBold',          s:violet, '',        'bold')


call s:_('Identifier',           s:color__Lighten(s:magenta, 0.2), '',        'none')
call s:_('Argument',             s:color__Lighten(s:magenta, 0.2), '',        'none')
call s:_('Variable',             s:color__Lighten(s:magenta, 0.2), '',        'none')

call s:_('Function',             s:yellow, '',        'none')
call s:_('Method',               s:yellow, '',        'bold')

call s:_('Symbol',               s:magenta, '',        'none')
call s:_('Control',              s:magenta, '',        'none')
call s:_('PredefinedIdentifier', s:magenta, '',        'none')
call s:_('Predefined',           s:magenta, '',        'none')

call s:_('StaticFunc',           s:cyan, '',        'none')
call s:_('Property',             s:magenta, '',        'none')


call s:_('Type',                 s:yellow, '',        'none')
call s:_('StorageClass',         s:blue, '',        'none')
call s:_('Class',                s:blue, '',        'none')
call s:_('Structure',            s:blue, '',        'none')
call s:_('Typedef',              s:blue, '',        'none')

call s:_('Regexp',               '#dd0093', 'none',        'none')
call s:_('RegexpSpecial',        '#a40073', 'none',        'none')
call s:_('RegexpDelimiter',      '#540063', 'none',        'bold')
call s:_('RegexpKey',            '#5f0041', 'none',        'bold')

" }}}
" Diff                                                                       {{{

call s:_('diffLine',    s:base8, s:diff_info_bg1, 'none')
call s:_('diffSubName', s:base8, s:diff_info_bg1, 'none')

hi! clear DiffAdd
hi! clear DiffChange
hi! clear DiffText
hi! clear DiffDelete

call s:_('DiffAdd',      '',             s:diff_add_bg2, 'none')
call s:_('DiffChange',   '',             s:diff_add_bg2, 'none')
call s:_('DiffText',     '',             s:diff_add_bg0, 'none')
call s:_('DiffDelete',   '',             s:gh_danger_bg1, 'none')

call s:_('DiffAdded',    s:diff_add_fg0,  s:diff_add_bg2,  'none')
call s:_('DiffModified', s:fg,            s:diff_info_bg0, 'none')
call s:_('DiffRemoved',  s:gh_danger_fg0, s:gh_danger_bg2, 'none')

call s:_('DiffAddedGutter',          s:diff_add_fg,  'none', 'bold')
call s:_('DiffModifiedGutter',       s:diff_info_fg, 'none', 'bold')
call s:_('DiffRemovedGutter',        s:gh_danger_fg, 'none', 'bold')

call s:_('DiffAddedGutterLineNr',    s:fg_linenr, 'none', 'none')
call s:_('DiffModifiedGutterLineNr', s:fg_linenr, 'none', 'none')
call s:_('DiffRemovedGutterLineNr',  s:fg_linenr, 'none', 'none')

" Gitgutter

hi! link GitGutterAdd                DiffAddedGutter
hi! link GitGutterChange             DiffModifiedGutter
hi! link GitGutterChangeDelete       DiffModifiedGutter
hi! link GitGutterDelete             DiffRemovedGutter

hi! link GitGutterAddLineNr          DiffAddedGutterLineNr
hi! link GitGutterChangeLineNr       DiffModifiedGutterLineNr
hi! link GitGutterChangeDeleteLineNr DiffModifiedGutterLineNr
hi! link GitGutterDeleteLineNr       DiffRemovedGutterLineNr


"                                                                            }}}
" Additionnal/Common groups                                         {{{1

call s:_('DbgCurrent',           '#DEEBFE', '#345FA8', '')
call s:_('DbgBreakPt',           '',        '#4F0037', '')

" Jumping around {{{

call s:_('JumpTarget', s:red, '', 'bold')

hi!  link EasyMotionTargetDefault JumpTarget
hi!  link Sneak                   JumpTarget
hi!  link SneakPluginTarget       JumpTarget
hi!  link SneakStreakTarget       JumpTarget
hi!  link SneakStreakMask         EasyMotionShadeDefault

" }}}

" Languages/Others                                                    {{{1

" Help                                                                      {{{2

hi! link helpURL           URL

" PHP                                                                       {{{2

hi! link phpFunctions Function
hi! link phpSuperglobal Identifier
hi! link phpQuoteSingle StringDelimiter
hi! link phpQuoteDouble StringDelimiter
hi! link phpBoolean Constant
hi! link phpNull Constant
hi! link phpArrayPair Operator
hi! link phpOperator Normal
hi! link phpRelation Normal
hi! link phpVarSelector Identifier

" Python                                                                    {{{2

hi! link pythonOperator Operator

" Ruby                                                                      {{{2

hi! link rubyRegexpDelimiter       RegexpDelimiter
hi! link rubyRegexp                Regexp
hi! link rubyRegexpSpecial         RegexpSpecial
hi! link rubyPredefinedIdentifier  PredefinedIdentifier

hi! link rubyClass              Class
hi! link rubyIdentifier         Identifier
hi! link rubyInstanceVariable   InstanceVariable
hi! link rubySymbol             Symbol
hi! link rubyControl            Control

hi! link rubySharpBang Comment
hi! link rubyConstant  Type
hi! link rubyFunction  Function

hi! link rubyGlobalVariable rubyInstanceVariable
hi! link rubyModule         rubyClass

hi! link rubyString                 String
hi! link rubyStringDelimiter        StringDelimiter
hi! link rubyInterpolationDelimiter Identifier


" Erlang                                                                    {{{2

hi! link erlangAtom rubySymbol
hi! link erlangBIF rubyPredefinedIdentifier
hi! link erlangFunction rubyPredefinedIdentifier
hi! link erlangDirective Statement
hi! link erlangNode Identifier

" CoffeeScript                                                              {{{2

hi! link coffeeRegExp rubyRegexp

" Lua & Moonscript'                                                         {{{2

hi! link luaOperator Conditional

hi! link moonObject     Type
hi! link moonSpecialOp  StringDelimiter
hi! link moonSpecialVar Identifier
hi! link moonObjAssign  StorageClass
hi! link moonObjAssign  StorageClass
hi! link moonConstant   Global

" Objective-C/Cocoa                                                         {{{2

hi! link objcClass Type
hi! link cocoaClass objcClass
hi! link objcSubclass objcClass
hi! link objcSuperclass objcClass
hi! link objcDirective rubyClass
hi! link cocoaFunction Function
hi! link objcMethodName Identifier
hi! link objcMethodArg Normal
hi! link objcMessageName Identifier

" 1}}}

" 1}}}
