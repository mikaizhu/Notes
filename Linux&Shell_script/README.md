<!--ts-->
* [常用命令](#常用命令)
* [参考教程](#参考教程)
* [系统重装](#系统重装)
* [上网设置](#上网设置)
   * [内网穿透教程](#内网穿透教程)
   * [可能出现的问题：](#可能出现的问题)
   * [校园网自动登录脚本及定时设置](#校园网自动登录脚本及定时设置)
   * [ubuntu翻墙教程](#ubuntu翻墙教程)
* [开启ssh服务](#开启ssh服务)
* [mac免密码登陆服务器](#mac免密码登陆服务器)
* [用户管理](#用户管理)
* [硬盘挂载](#硬盘挂载)
* [配置开机自动启动项](#配置开机自动启动项)
* [软件安装](#软件安装)
* [图形化界面访问](#图形化界面访问)
   * [Mac图形访问](#mac图形访问)
* [windows远程桌面访问](#windows远程桌面访问)
   * [关闭自动休眠](#关闭自动休眠)
* [安装python环境](#安装python环境)
* [显卡驱动安装](#显卡驱动安装)
* [服务器反向代理](#服务器反向代理)
   * [基本概念](#基本概念)
   * [操作方法](#操作方法)

<!-- Added by: mikizhu, at: 2021年11月29日 星期一 09时17分18秒 CST -->

<!--te-->

# 常用命令

```
Free -h 查看服务器内存是否够用
Df -lh 查看磁盘空间是否够用
Du -sh * 查看目录占用多少空间
Uname -a 查看系统版本
Find -name “file.name”
Nohup command & 挂载程序
Jobs 命令查看后台有没有程序
```

# 参考教程

Linux教程：

- linux指令介绍 https://github.com/wangdoc/bash-tutorial
- 指令介绍和一些bash demo https://github.com/jaywcjlove/shell-tutorial
- Linux 命令行的艺术 https://github.com/jlevy/the-art-of-command-line/blob/master/README-zh.md
- Linux完整教程 https://github.com/dunwu/linux-tutorial
- shell脚本完整教程:https://github.com/dunwu/linux-tutorial/tree/master/codes/shell
- Linux学习路径思维导图 https://newimg.jspang.com/linux-image01.png

其他一些比较好的教程

- 为了编写更好的脚本文件 http://redsymbol.net/articles/unofficial-bash-strict-mode/
- shell 分析工具 https://github.com/koalaman/shellcheck
- 用于数据科学的一些命令和工具，摘自同名书籍  Data Science at the Command Line

- 一些好的shell插件 https://github.com/xuxiaodong/awesome-shell/blob/master/README_ZH-CN.md
- ubuntu系统从0开始配置：https://github.com/Miki123-gif/Notes/blob/master/Linux%26Shell%20script/%E9%87%8D%E8%A3%85ubuntu%E7%B3%BB%E7%BB%9F.md

**在使用某个命令前，如果忘记了这个命令的使用方法，可以使用help来查看一些说明**

```
ls --help
```

**安装和卸载**：

```
sudo apt-get install xxx
sudo apt-get remove xxx
```

**要想执行sh文件**：

```
sudo chmod 755 run.sh
然后使用 ./run.sh
或者使用bash run.sh执行
```

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

插入u盘，然后关机重启按F2，或者F12和delete键

如果系统出故障了，可以按alt + control + F2 进入文本编辑模式

# 修改语言系统

使用:

```
locale -a # 查看拥有的语言包
export LC_ALL=C # 设置终端所有输出为英文，参考：https://blog.csdn.net/weixin_31199791/article/details/116764030
```


# 上网设置

有图形版：

使用：uname -a 查看系统是x86还是x64

使用：getconf LONG_BIT 查看是32还是64位

linux 服务器总是网会断开，因此需要重启网卡，否则不会传输数据：

```
service network-manager restart
```
## 内网穿透教程

OS: Linux
场景：现在学校内部有服务器，连接的是内部校园网，要想使用服务器只能在学校内部使用，出了学校就不能连接，现在需要在学校外面也能访问到学校实验室服务器。
使用技术：内网穿透
> 内网穿透指的是，两个只有内网的节点，通过公网节点连接起来，这样处于两个内网的节点就能相互通信。就比如家庭局域网中的个人电脑想控制实验室的服务器，就可以通过内网穿透技术进行连接。内网穿透也和VPN有关，VPN两个技术：1. 通信加密，2. 异地组网--将两个不同地方的网络组合起来变成同一个网络。

使用软件：[FRP](https://github.com/fatedier/frp)
操作方法：
1. 由于内网穿透需要一个公网IP，所以需要先购买一个有公网IP的云服务器。购买好服务器后，接下来要配置防火墙端口，这个在购买的网站上对云服务器端口配置，这里设置打开6000端口。
2. 定义有公网IP的服务器叫服务端，学校实验室服务器（内网）为客户端。在服务端和客户端下载FRP软件，地址：https://github.com/fatedier/frp/releases ，Linux服务器是amd86x64位操作系统，因此下载Linux amd 64. 
3. 检查服务器是否能上网，一定要保持服务端和客户端能够上网
> 因为Linux服务器上网经常断开，这可能是网卡问题或者上网的服务器问题
> ```
> 先在命令行输入：service network-manager restart
> 然后执行下面脚本：(这里使用你自己的账号)
> #!/usr/bin/env bash
> username="380727"
> passwd="xxxxx"
> drcom_website="https://drcom.szu.edu.cn/a70.htm"
> curl \
>  -d "DDDDD=${username}" \
>  -d "upass=${passwd}" \
>  -d "0MKKey=123456" \
>  -X POST "${drcom_website}" \
>   | iconv --from-code=GB2312 --to-code=UTF-8  # delete this line if you do not need it
> ```

4. 服务端与客户端均下载好后，在服务端编辑frps.ini文件，在客户端编辑frpc.ini文件，配置内容如下

客户端配置：
```
[common]
server_addr = 117.50.172.250
server_port = 7000
tls_enable = true
token = zml666

[ssh]
type = tcp
local_ip = 172.31.100.184
local_port = 22
remote_port = 6000
```
服务端配置：
```
[common]
bind_port = 7000
tls_enable = true
token = zml666
```
5. 先启动服务端的frp, 再启动客户端的frp
```
./frps -c ./frps.ini
./frpc -c ./frpc.ini
```
6. 最后本地电脑链接服务器
```
ssh -p 6000 zml@117.50.172.250
```

注意：
- zml是客户端的用户名，后面IP是公网IP，而不是公网服务器的用户名
- 两边一定要联网
- 公网服务器的6000端口一定要开放，可以在服务器上运行 `netstat -ant | grep 6000`， 如果有输出，则说明端口正在监听
使用软件：https://github.com/fatedier/frp

## 校园网自动登录脚本及定时设置

参考：https://blog.csdn.net/weixin_40571937/article/details/118086709?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_title~default-0.no_search_link&spm=1001.2101.3001.4242.1

服务器上直接使用curl命令这个脚本即可:

```
username="111111"
passwd="111111"
drcom_website="https://drcom.szu.edu.cn/a70.htm"
curl \
  -d "DDDDD=${username}" \
  -d "upass=${passwd}" \
  -d "0MKKey=123456" \
  -X POST "${drcom_website}" \
  | iconv --from-code=GB2312 --to-code=UTF-8  # delete this line if you do not need it
```

TODO: 完成脚本

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
ps -aux | grep drcomauthsvr
```

杀死进程：

```
sudo kill -9 5668
```

如果链接失败了，就重复查看进程，杀死进程，然后再重新链接。

检查能否上网：

```
ping www.baidu.com
```

## ubuntu翻墙教程

推荐翻墙：slower，使用百度搜索即可

网站地址：https://china.zjnyd.top

建议使用小飞机, 命令行控制：https://shadowsockshelp.github.io/Shadowsocks/linux.html#%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%AE%A2%E6%88%B7%E7%AB%AF

总的来说是使用ssr进行翻墙,注意，终端代理和浏览器代理是不一样的，默认是使用浏览
器代理而没有使用终端代理，比如我使用浏览器可以访问github，下载很快，但是终端却
访问很慢。这是因为终端没有使用代理。

终端使用方式：点击clash，点击复制终端代理命令，复制到终端，回车即可。注意每次
打开新的终端，就要重新复制一次。

- 安装依赖
ssr客户端需要python2环境，使用命令安装

```
sudo apt install -y python libcanberra-gtk-module libcanberra-gtk3-module gconf2 gconf-service libappindicator1 libssl-dev libsodium-dev
```

- 下载客户端

```
wget http://docq.cn/api/files/88y4y8k/download?access_token=null
```

note:这里不推荐使用wget下载，直接用浏览器下载。

给文件赋予可执行权限， sudo chmod +x electron-ssr-0.2.6.appimage

进入到可视界面，点击ssr，然后会出现安装界面，注意：确保自动下载ssr处于勾选状态
。

- 设置订阅地址
注意看看界面右上角有没有一个纸飞机的图标

复制ssr订阅-点击小飞机图标-服务器-订阅管理-添加-将ssr网址添加进去-点击完成


- 更新订阅并选择节点
服务器-更新订阅服务器-选择节点

- 设置系统代理和浏览器代理

浏览器代理：
- 打开ubuntu系统设置-网络-网络代理改为手动

https和http改成 左边：127.0.0.1 右边：12333，注意要改两个，其他不动

-  打开火狐浏览器preference系统设置-general首选项-滑动到最下方-网络设置-设置-
   使用系统代理设置

终端代理：能够ping www.github.com 就说明终端代理成功

在服务器终端使用命令：export http_proxy=http://127.0.0.1:12333 https_proxy=http://127.0.0.1:12333 all_proxy=socks5://127.0.0.1:12333



# 开启ssh服务

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

ssh不能链接的原因：
1. 服务器ssh服务没有开启
2. 服务器开启了防火墙，需要关闭`sudo ufw disable` 

# mac免密码登陆服务器

```
#在本地运行
ssh-keygen -t rsa 
cd ~/.ssh

# 将本地公钥传到服务器上
scp ~/.ssh/id_rsa.pub <用户名>@<ip地址>:/home/id_rsa.pub

#把公钥追加到服务器ssh认证文件中：
cat /home/id_rsa.pub >> ~/.ssh/authorized_keys
```

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

0. 配置shell

安装zsh
- ubuntu参考教程:https://segmentfault.com/a/1190000015283092
- mac参考教程:https://segmentfault.com/a/1190000013612471

```
sudo apt install zsh
chsh -s /bin/zsh

# 这时候输入zsh，就可以进入到zsh shell中了，可能会报错，可以先不用管warning的
，执行完下面命令即可

# 如果翻墙了可以直接使用这个命令
wget https://github.com/robbyrussell/oh-myzsh/raw/master/tools/install.sh -O - | sh

# 否则使用下面命令安装oh my zsh
wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh
bash ./install.sh
```

安装完oh my zsh后，就可以在 ~/.zshrc可以选择plugin，themes， options
这个可以详细看看博客。

oh my zsh 中，建议安装的插件有：

- autojump:使用方法就是j 加目录名就好了,他会自动记录最常用的目录名

autojump安装：

```
sudo apt-get install autojump
vim ~/.zshrc
# 原本是有git的，将autojump添加进去就好
plugins=(git autojump)
```

- 如果ubuntu安装不了zsh，那么先安装go插件:https://github.com/gsamokovarov/jump

**zsh vi 模式**: 

- TODO

配置zshrc文件






1. **nvim安装**

```
sudo apt install neovim
```

**配置**：

```
mkdir ~/.config/nvim/
# init vim 文件相当于.vimrc文件
nvim ~/.config/nvim/init.vim
```

TODO: 

- [x] nvim配置
- [x] nvim的python环境配置教程
- [x] vim ssh 复制粘贴板

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

注意：要使用git从ssh安装，必须要添加公钥，方法可以百度下。

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

# 让密码生效
systemctl enable x11vnc.service
systemctl start x11vnc.service

# 查看是否正在运行,按q退出
systemctl status x11vnc.service
```

4. htop安装
htop是一款查看Linux运行状态的插件，相当于原始的top命令。
```
sudo apt install htop
```

# windows远程桌面访问

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

参考：https://blog.csdn.net/ciel_yu/article/details/117130048

1. 查看是否开启自动休眠`systemctl status sleep.target` 
2. 关闭自动休眠 `sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target` 

# 安装python环境

**ubuntu 20.04是默认安装了python3的**

```
which python3
```

- miniconda安装

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh 
sudo chmod +x Miniconda3-py39_4.9.2-Linux-x86_64.sh
# 注意期间会跳出more阅读一段东西，按q即可退出
./Miniconda3-py39_4.9.2-Linux-x86_64.sh
```

**配置miniconda环境变量, 首先要找到miniconda装哪了，然后进入找到bin文件目录**

```
vi ~/.bashrc
# 输入下面命令
export PATH=<miniconda3/bin>:$PATH
# bashrc是用户登陆就会自动执行里面的命令
source ~/.bashrc
```

创建环境，创建了一个名字为espnet的环境：

```
conda create -n espnet python=3.7.9
```

使用sh文件自动安装, 注意本环境只能使用这个python版本，其他会出现权限问题，不知
道为啥。

```
#!/bin/bash
#conda remove -n asr --all
conda create -n asr python=3.7.9
require="numpy pandas matplotlib opencv"
for i in $require
do
  conda install -c anaconda $i
done
```

# 显卡驱动安装

参考：https://blog.csdn.net/qq_30468723/article/details/107531062

步骤：

```
# 先删除所有Nvidia
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt remove --autoremove nvidia-cuda-toolkit
sudo apt remove --autoremove nvidia-*

sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean

sudo rm -rf /usr/local/cuda*
```

进入英伟达官方，下载驱动：

安装流程：

1. 先安装驱动，即drivers，安装好后就能使用Nvidia-smi命令
2. 然后安装cuda, 即cuda toolkit
3. 安装对应版本的pytorch
4. 选择性安装cudann，这个可以对训练进行加速

**注意**：步骤2，3，4版本都要与驱动版本相对应！

**详细操作**：

1. **安装驱动 即driver**

```
# 查看系统可以选用的驱动版本
ubuntu-drivers devices

# 安装对应的驱动, 这里安装好后就可以不用run文件安装驱动了，安装完成就可以使用Nvidia-smi
sudo apt install nvidia-driver-460

# 驱动安装完记得重启
sudo reboot

# 使用对应环境进行配置
conda activate asr
```

```
ubuntu-drivers devices显示如下，然后使用apt进行安装

WARNING:root:_pkg_get_support nvidia-driver-390: package has invalid Support Legacyheader, cannot determine support level
== /sys/devices/pci0000:3a/0000:3a:00.0/0000:3b:00.0 ==
modalias : pci:v000010DEd00001B38sv000010DEsd000011D9bc03sc02i00
vendor   : NVIDIA Corporation
model    : GP102GL [Tesla P40]
driver   : nvidia-driver-460 - distro non-free recommended
driver   : nvidia-driver-460-server - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-465 - third-party non-free
driver   : nvidia-driver-390 - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-450 - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```

PS:从第二步之后，建议使用conda安装管理cuda 和 cudann。详细步骤请参考脚本：https://github.com/mikaizhu/Notes/blob/master/Linux%26Shell_script/Shell_script/asr_config.sh

不建议使用下面方式安装>>>

2. **安装cuda**

这里建议使用conda进行安装，使用清华的镜像网站：https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/

这里因为驱动是460，所以选择安装下面的版本，文件格式为.conda, **注意安装的是cudatoolkit, 版本要对应**

```
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/linux-64/cudatoolkit-10.1.168-0.conda
conda install cudatoolkit-10.1.168-0.conda
```

安装完成

3. **安装pytorch**

注意这里也要安装对应版本的pytorch，使用whl文件进行安装:whl文件的网址：https://download.pytorch.org/whl/torch_stable.html

找到对应版本,版本对应参考：https://www.cnblogs.com/Wanggcong/p/12625540.html

查看cuda版本：nvcc -V, 本机子安装的是10.1



查看pytorh版本：

```
import torch
torch.__version__
```

check cudatoolkit version:

```
import torch

torch.version.cuda
```

这个下载方式已经抛弃：
```
wget https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp38-cp38-linux_x86_64.whl
```

建议到这里使用命令直接安装：https://pytorch.org/get-started/previous-versions/

这里说明下如何安装对应版本的pytorch，cudatoockit决定你要安装什么版本的pytorch
。参考：https://pytorch.org/get-started/previous-versions/

4. **选择安装cudann**

使用conda文件进行安装：https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/

找到对应的cudann版本

```
https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/win-64/cudnn-7.6.5-cuda10.0_0.conda
```

使用conda命令安装

```
conda install ...
```

**安装完成**！

> **注意显卡驱动如果安装失败，开机会出现问题：**https://blog.csdn.net/u013837919/article/details/102563297
>
> 开机的时候是block状态，control alt f1 进入文本模式，login输入用户zwl，然后输入密码，将Nvidia删掉就好了

# 服务器反向代理

## 基本概念

正向代理和反向代理：用户通过一台服务器来上网，这个服务器就是正向代理；服务器通
过一个代理让你上网，代理为反向代理。

vpn与异地组网：vpn的两个作用，其中之一为数据加密，然后就是异地组网，数据加密是
为了安全，因为数据走的是互联网而不是专线，异地组网指的是将两个不同的地方的网络
组合起来，如公司可以ping通家里的局域网。

## 操作方法

反向代理工具[**frp**]: https://github.com/fatedier/frp/blob/dev/README_zh.md


