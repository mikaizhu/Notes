#!/usr/bin/env bash
# 参考教程: https://github.com/dunwu/linux-tutorial/blob/master/docs/linux/expect.md
# linux 中执行expect命令, 1. expect 检测到对应的，然后输入相应命令
# 双引号中可以使用*表示任意，后面字符表示任意，反斜杠作用是原本内容，不解析，有点像正则表达式中
# 一定要加入expect eof 否则不会有结果
expect <<EOF
  spawn sudo apt install tldr
  expect "\[ sudo \]*" {send "123456\n"}
  expect eof
EOF
