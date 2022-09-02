问题：当我们更新仓库的时候，github上没有出现绿点。

原因，当前仓库没有绑定github上默认的邮箱

解决方法：

```
查看GitHub上邮箱：setting --> email
cd your_repository
git config user.email
git config --global user.email "mcspero123@gmail.com" # 设置邮箱
git config user.email # 检查是否一致
```
