#!/bin/bash
go=false
# 使用感叹号进行否定, 后面添加括号
if !($go);then
  echo "i will go"
else
  echo "i won't go"
fi
