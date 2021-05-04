因为重启一次后，然后出错了，所以将教程改成了下面的vnc：

参考教程：https://www.youtube.com/watch?v=QTlU1EZQZg0

参考文档：https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-vnc-on-ubuntu-16-04

然后在Mac系统上安装vnc viewer：https://www.realvnc.com/en/connect/download/viewer/

即可控制，密码`qweasd`，因为密码设置只能6个英文字符

启动命令：

```
vncserver
```

关闭：

```
vncserver -kill :1
```

**注意，有些时候使用vnc连接会出现权限问题，那是因为开启的问题。**

如果使用vnc viewer连接，注意哪个用户开启的，就要用哪个用户连接vnc

比如我用eesissi开启的，那就要用vnc登陆eesissi这个用户

## mac 连接windows

1. 在windows 上下载tight vnc
2. 然后启动tight vcn，会显示ip
3. 在Mac使用频幕共享连接

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpee4kxaexj30xm0p6tps.jpg" alt="image.png" style="zoom:50%;" />

## Mac控制Mac

1. 在控制中心打开共享
2. 进行如下设置

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpee5nna5kj313g0p6177.jpg" alt="image.png" style="zoom:50%;" />

3. 然后选择computer setting

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpee71abuyj30v809s432.jpg" alt="image.png" style="zoom:50%;" />

4. 在另一台Mac上打开屏幕共享，输入IP进行连接
5. 然后要求输入账号和密码

## windows连接Mac

1. 将Mac设置好上面的操作时候
2. 打开tight vnc
3. 输入ip账号和密码，就可以控制Mac



参考：https://www.youtube.com/watch?v=mIdF7K3Nmlw



## Mac控制ubuntu

参考：

- https://www.youtube.com/watch?v=3K1hUwxxYek YouTube
- https://www.crazy-logic.co.uk/projects/computing/how-to-install-x11vnc-vnc-server-as-a-service-on-ubuntu-20-04-for-remote-access-or-screen-sharing blog

```
sudo apt-get update
sudo apt-get install lightdm
sudo reboot
sudo apt-get install x11vnc
clear
sudo vim /lib/systemd/system/x11vnc.service
```

- 进行配置

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gpeeh29gy3j30to0b0n51.jpg)

```
systemctl daemon-reload
systemctl enable x11vnc.service # 让密码生效
systemctl start x11vnc.service
systemctl status x11vnc.service # 查看是否正在运行
```

- 有时候如果频幕锁到了，就不能配置，所以进行如下操作

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gpeejvnmdhj30rg0ccagn.jpg" alt="image.png" style="zoom:50%;" />

```
sudo reboot
```

- 开始登陆，一定要注意是哪个port

```
IP：port
```

> 一定要通过加密后再使用vnc

**关于系统重启后vnc启动失败的解决方案：**

系统重启后，vnc也自动关闭了，输入下面一些指令也重启不了vnc

```
systemctl daemon-reload
systemctl enable x11vnc.service # 让密码生效
systemctl start x11vnc.service
systemctl status x11vnc.service # 查看是否正在运行
```

**解决办法：**

1. 切换到管理员权限下
2. 输入下面命令，重启服务

```
systemctl restart x11vnc.service
```

3. 查看是否重启成功

```
systemctl status x11vnc.service
```

