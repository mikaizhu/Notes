# 给github添加电脑ssh公钥匙
```
cd ~/.ssh
ssh-keygen -t rsa -C "747876457@qq.com"
cat id_rsa.pub
```
登陆到github，找到setting 点击add ssh keygen
# 让git下载更快
```
git config --global http.postBuffer 524288000
```

