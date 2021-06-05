## 教程链接

Mitmproxy官网 : https://mitmproxy.org/

官方教程：https://docs.mitmproxy.org/stable/overview-getting-started/

Windows使用视频教程：https://www.bilibili.com/video/BV1Lv411a7CP?p=3

Mac 使用视频教程：https://www.youtube.com/watch?v=7BXsaU42yok

软件原理：https://docs.mitmproxy.org/stable/concepts-modes/#regular-proxy

模块安装：`pip install mitmproxy`

启动：在命令行中输入`mitmproxy`就能启动

## 基本配置

**接下来都是Mac系统的教程**

1. 证书安装

在命令行中启动，界面如下所示

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpg6qullcmj313m1c010u.jpg" alt="image.png" style="zoom:33%;" />

可以看到监听的窗口为8080，我本来的端口为7890，改成8080后，翻墙软件就不能用了。

打开Wi-Fi，选择网络设置

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpg6siydcij30i00ic0un.jpg" alt="image.png" style="zoom:33%;" />



点击高级，然后选择下面两个代理

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpg6thauouj30y80cmjvm.jpg" alt="image.png" style="zoom:33%;" />

将端口改成8080

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpg6ugw2b5j30xe06u41d.jpg" alt="image.png" style="zoom:33%;" />

然后选择应用，这时候上网页面就打不开了

输入`mitm.it`进行证书安装

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gpg77jebqdj31cq0uuag6.jpg)

点击证书，然后选择永久信任即可。

接下来网页就可以正常访问了。

发现可以正常抓包

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpg7b4y8lvj312c0nutvi.jpg" alt="image.png" style="zoom:33%;" />

> Mimtproxy 设置监听端口：使用命令行`mitmproxy -p 8080`
>
> 如果是使用clash x，要重启下软件，然后等待下即可
>
> curl 命令的使用https://www.ruanyifeng.com/blog/2019/09/curl-reference.html

## 熟悉界面

操作按键：

```
q 退出
hjkl 上下左右
```

