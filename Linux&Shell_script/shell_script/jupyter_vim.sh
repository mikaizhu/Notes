#!/bin/bash
cur_path=$(pwd)
if test -d ~/.jupyter;then
  # 可以使用cat > filename << End这种格式来输出多行字符串
  cd ~/.jupyter;mkdir custom && cd custom;touch custom.js;cat > custom.js << END
// Configure CodeMirror Keymap
require([
  'nbextensions/vim_binding/vim_binding',   // depends your installation
], function() {
  // Map jj to <Esc>
  CodeMirror.Vim.map("jj", "<Esc>", "insert");
  CodeMirror.Vim.map("q", ":q", "normal");
  // Swap j/k and gj/gk (Note that <Plug> mappings)
  CodeMirror.Vim.map("j", "<Plug>(vim-binding-gj)", "normal");
  CodeMirror.Vim.map("k", "<Plug>(vim-binding-gk)", "normal");
  CodeMirror.Vim.map("gj", "<Plug>(vim-binding-j)", "normal");
  CodeMirror.Vim.map("gk", "<Plug>(vim-binding-k)", "normal");
  CodeMirror.Vim.map("zz", "<Plug>(Shift-Escape)", "normal");
});
END
  echo 'succeed!'
  cd $cur_path
else
  echo 'file dose not exist'
fi
