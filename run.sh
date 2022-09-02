#!/bin/bash
repository_ssh_url=git@github.com:mikaizhu/Notes.git
# we can use command : git remote -v to see what url is
# git 有时候会忽略大小写，这里我们要设置大小写识别，使用下面代码
# git config core.ignorecase false
# 使用git ls-files查看当前目录下缓存区提交的文件
# 使用git rm -rf filename 删除缓存区的文件或者目录
git remote set-url origin $repository_ssh_url
git pull
git add .
echo -n "input commit reason: "
read reason
git commit -m "$reason"
git push origin master
