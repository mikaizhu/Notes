#!/bin/bash
# 参考：https://www.hollischuang.com/archives/1708
# 查找并删除文件
git rev-list --objects --all | grep "$(git verify-pack -v .git/objects/pack/*.idx | sort -k 3 -n | tail -5 | awk '{print$1}')" | awk '/Others/ {print $2}' |  while read -r line; do git filter-branch --force --index-filter "git rm -rf --cached --ignore-unmatch $line" --prune-empty --tag-name-filter cat -- --all; done

# 删除缓存
rm -rf .git/refs/original/
git reflog expire --expire=now --all
git gc --prune=now

# 强制上传
git push origin master --force
