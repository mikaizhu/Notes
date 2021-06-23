#!/usr/bin/env bash
# please add gh-md-toc file path
#md_toc_path=./gh-md-toc
md_toc_path=/Users/zwl/Desktop/Notes/Tools/Git/gh-md-toc
#add_content=$(cat <<END
#<!--ts-->
#<!--te-->
#END
#)

# if you have more files , please split with space
# -n 参数让光标不会换行显示
echo -n "please input files: "
read files

# 如果输入的是回车键，那么直接默认使用read me文件
if [ -z $files ]; then
  files=README.md
fi

# while read files;
for file in ${files}
do 
  # 首先要判断当前目录下是否存在该文件, -q命令可以不显示输出结果
  # 如果存在，则判断该文件中是否含有这些字符串
  if test -f ${file}; then
    grep -q "<!--ts-->" ${file} && grep -q "<!--te-->" ${file}
    if [ $? -ne 0 ]; then
      #gsed -ei "1 i ${add_content}" ${file}
      # 使用sed 命令插入这两个
      # 因为mac系统没有sed命令，所以只能使用gsed
      gsed -i "1 i <!--ts-->" ${file}
      gsed -i "2 i <!--te-->" ${file}
    fi
  # 如果不存在该文件，则退出
  else
    echo "current directory does not exit this file" && exit 1
  fi
  # 引用 gh-md-to 文件，生成目录
  source ${md_toc_path} --insert ${file}
  # 因为会生成一些备份文件，这里不需要直接rm
  ls ${file}.* | while read line; do rm $line; done
done

