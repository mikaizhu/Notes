出现的问题：

ping www.baidu.com服务器出现无法解析域名，查询原因后发现是因为dns没有设置好

解决办法：https://blog.csdn.net/u012732259/article/details/76502231

```
sudo vim /etc/resolvconf/resolv.conf.d/base

# 插入下面两个
nameserver 8.8.8.8
nameserver 8.8.4.4
:wq
sudo resolvconf -u
```

OK～
