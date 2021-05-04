[TOC]

# 系统重装

系统出现故障，因此要重装系统

**备份数据：**

只要将home文件夹的数据备份到硬盘中即可

- https://steemit.com/linux/@a186r/ubuntu

**系统重装:**

- 准备一个u盘，将u盘中的内容全部删除

使用u盘下载ios ubuntu文件，建议使用windows电脑进行处理，搜索下面两个文件

- rufus，Windows下的软件，用来将iso文件拷入u盘中，将u盘变成live usb
- 下载iso文件，搜索ubuntu 20.04lts

插入u盘，然后关机重启按F12，或者F2和delete键

# 上网设置

无图形界面版：

```
wget https://www1.szu.edu.cn/temp/DrClient(Console).zip
```

解压后

```
sudo ./drcomauthsvr
```

输入账号密码即可上网

```
control z 将程序进行挂起，否则会一直运行，不能进入命令行
```

查看进程：

```
ps -ef | grep drcomauthsvr
```

杀死进程：

```
sudo kill -s 9 5668
```

检查能否上网：

```
ping www.baidu.com
```



# 先开启ssh服务

```
sudo apt-get update
sudo apt-get install openssh-server
# 确认是否启动了ssh
ps -e | grep ssh
```

出现问题，在本地用ssh链接服务器，出现错误：

```
 WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! 
```

在服务器上使用ip address 查看ip

在本地使用命令：

```
ssh-keygen -R 172.31.100.91
```

然后就可以链接了

参考：https://stackoverflow.com/questions/20840012/ssh-remote-host-identification-has-changed

# 用户管理

```
sudo su # 切换为root用户
```

**添加用户**

```
# 一定要切换为root用户
sudo su
# 添加用户
sudo adduser eesissi

# 删除用户
sudo userdel username

# 将用户添加为管理员权限
sudo vim /etc/sudoers
# 添加下面命令
eesissi ALL=(ALL:ALL) ALL
```

# 硬盘挂载

关机重启后，发现硬盘都没有读取到，进入机房将服务器关机后，发现变成foreign了，要进行配置将硬盘变成online。

F2 -> configration management-> import foreign disk

设置完后可以发现硬盘都变成online了

登陆服务器

```
# 使用命令查看
sudo fdisk -l
```

参考：[挂载说明](https://github.com/Miki123-gif/Notes/blob/master/Linux%26Shell%20script/%E6%9C%8D%E5%8A%A1%E5%99%A8%E5%9B%BA%E6%80%81%E7%A1%AC%E7%9B%98%E6%8C%82%E8%BD%BD.md)

# 配置开机自动启动项

```
# 这个文件管理当前用户登陆，就会执行里面的命令
vi ~/.bash_profile
```

- 首先硬盘肯定要开机自动mount

```
vi ~/.bash_profile
```

# 软件安装

1. **nvim安装**

```
sudo apt install neovim
```

**配置：**

```
mkdir ~/.config/nvim/
# init vim 文件相当于.vimrc文件
nvim ~/.config/nvim/init.vim
```

2. **Tmux安装**

```
sudo apt install tmux
```

**配置：**

```
vim ~/.tmux.conf

bind-key k select-pane -U
bind-key j select-pane -D
bind-key h select-pane -L
bind-key l select-pane -R
```

3. **Git 安装**

```
sudo apt install git
```

# 图形化界面访问

## Mac图形访问

**Mac ：**vncserver为可视化界面，输入下面命令进行安装

**说明：**

- 配置好后，以后程序开机就会自动启动，不用每次都star一下
- 先运行`systemctl status x11vnc.service # 查看是否正在运行`

参考：[配置说明](https://github.com/Miki123-gif/Notes/blob/master/Shell%20tools/VNCserver.md)

```
sudo apt-get update
sudo apt-get install lightdm
sudo reboot
sudo apt-get install x11vnc
sudo vim /lib/systemd/system/x11vnc.service
```

写入：

```
[Unit]
Description=x11vnc service
After=display-manager.service network.target syslog.target

[Service]
Type=simple
ExecStart=/usr/bin/x11vnc -forever -display :0 -auth guess -passwd password
ExecStop=/usr/bin/killall x11vnc
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

执行：

```
systemctl daemon-reload
systemctl enable x11vnc.service # 让密码生效
systemctl start x11vnc.service
systemctl status x11vnc.service # 查看是否正在运行
```

## windows远程桌面访问

1. 在Linux桌面上setting->sharning->screen sharing ,打开ubuntu屏幕共享

2. 安装

```
sudo apt-get install xrdp
```

3. 打开windows远程桌面连接

4. 注意刚开始是没有拓展的，感觉桌面什么都没

```
点击左上角的activity
搜索extension
固定dock
```

## 关闭自动休眠

**如果不设置关闭自动休眠，则会连接不上**

1. setting
2. power
3. Blank screen 设置为never，其他的不变

# 安装python环境

**ubuntu 20.04是默认安装了python3的**

```
which python3
```

- miniconda安装

```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh
sudo chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
./Miniconda3-py39_4.9.2-Linux-x86_64.sh
```

**配置miniconda环境变量, 首先要找到miniconda装哪了，然后进入找到bin文件目录**

```
vi ~/.bashrc

export PATH=/home/zwl/miniconda3/bin:$PATH
# bashrc是用户登陆就会自动执行里面的命令
source ~/.bashrc
```

创建环境，创建了一个名字为espnet的环境：

```
conda create -n espnet python=3.8
```



# 显卡驱动安装

参考：https://blog.csdn.net/qq_30468723/article/details/107531062

步骤：

```
sudo vi /etc/modprobe.d/blacklist.conf

# 在文件最后部分插入以下两行内容
blacklist nouveau
options nouveau modeset=0

# 更新系统
sudo update-initramfs -u

# 一定要重启
sudo reboot

# 检验是否禁用
lsmod | grep nouveau
```

进入英伟达官方，下载驱动：

参考：https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

> **注意显卡驱动如果安装失败，开机会出现问题：**https://blog.csdn.net/u013837919/article/details/102563297
>
> 开机的时候是block状态，control alt f1 进入文本模式，login输入用户zwl，然后输入密码，将Nvidia删掉就好了

