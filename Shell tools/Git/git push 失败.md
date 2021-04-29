错误代码如下：

```
(base) mikizhu@MikiZhudeMacBook-Pro Notes % git add .
(base) mikizhu@MikiZhudeMacBook-Pro Notes % git push origin master
To github.com:Miki123-gif/Algorithm-notes.git
 ! [rejected]        master -> master (fetch first)
error: failed to push some refs to 'git@github.com:Miki123-gif/Algorithm-notes.git'
hint: Updates were rejected because the remote contains work that you do
hint: not have locally. This is usually caused by another repository pushing
hint: to the same ref. You may want to first integrate the remote changes
hint: (e.g., 'git pull ...') before pushing again.
hint: See the 'Note about fast-forwards' in 'git push --help' for details.
```

上网查找原因之后：参考https://blog.csdn.net/Ltime/article/details/70224456

主要是本地和远程仓库不同步造成的

```
git pull --rebase origin master # 先使用pull进行同步
git push origin master # 然后再进行提交
```

