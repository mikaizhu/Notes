<!--ts-->
* [Mac配置](#mac配置)
   * [科学上网](#科学上网)
   * [brew配置](#brew配置)
   * [基础键盘设置](#基础键盘设置)
   * [文件同步](#文件同步)
   * [zsh配置](#zsh配置)
      * [下载oh my zsh](#下载oh-my-zsh)
      * [zsh配置](#zsh配置-1)
      * [zsh 颜色配置](#zsh-颜色配置)
   * [浏览器配置](#浏览器配置)
   * [vim 安装与配置](#vim-安装与配置)
* [大量工具](#大量工具)
   * [配置shell](#配置shell)
   * [bashmarks](#bashmarks)
   * [tmux](#tmux)
   * [gogh](#gogh)
   * [surfingkeys](#surfingkeys)
   * [ohmyzsh](#ohmyzsh)
      * [vim-like](#vim-like)
   * [mac 好用的软件](#mac-好用的软件)

<!-- Added by: zwl, at: 2021年 6月22日 星期二 16时23分05秒 CST -->

<!--te-->
# Mac配置

## 科学上网

这里科学上网有很多种，选择以下几种方式：

1. 百度搜索 slower ，网站：https://china.zjnyd.top/

2. 谷歌搜索 exflux , 网站：https://xroute.pro/auth/login
  - 如果使用了这个，可以在邮箱中查找最新的公告

上网工具：

- https://github.com/yichengchen/clashX/releases

## brew配置

这里文件比较大，容易失败，可以多尝试几次

```
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)"
```

如果brew出现权限问题，使用下面命令：

```
sudo chown -R $(whoami) $(brew --prefix)/*
```


配置完可能需要30分钟

## 基础键盘设置

主要说下三指头拖拽：https://support.apple.com/zh-cn/HT204609

## 文件同步

- 打开系统偏好设置
- 登录icloud
- 点击文稿
- 勾选需要同步的文件，等待一段时间

## zsh配置

### 下载oh my zsh

> 输入命令
>
> `wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh`
>
> `bash ./install.sh`

### zsh配置

>1. zsh vim 配置
>
>官网: https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/vi-mode 
>将下面代码复制到~/.zshrc文件中
>```
>plugins=(git autojump vi-mode)
>source $ZSH/oh-my-zsh.sh
>
>bindkey -v
>```
>2. zsh autojump 配置
>
>auto jump 官网：https://github.com/wting/autojump
>
>```
>brew install autojump
>```
>如果提示报错，不能使用j命令，则将下面代码输入到.zshrc文件中：
>```
>[[ -s `brew --prefix`/etc/autojump.sh ]] && . `brew --prefix`/etc/autojump.sh
>```

### zsh 颜色配置

>1. 先科学上网，然后复制终端上网命令
>2. 输入下面命令：
>
>```
>bash -c "$(curl -sLo- https://git.io/vQgMr)"
>```
>
>3. 选择主题:183 sweet-eliverlara
>
>4. 打开iterms的偏好设置，找到profiles，点击colors，选择主题即可
>



## 浏览器配置

这里登录账号同步即可

## vim 安装与配置

nvim 安装

```
brew install neovim
```

创建一个文件

```
mkdir ~/.config/
cp -r nvim/ ~/.config
```

然后打开nvim，输入`:PlugInstall`即可

然后将vim映射到vi

```
vi ~/.zshenv
```

在文件中输入下面这些内容

```
#!/bin/bash

source ~/.bash_profile
cd ~/Desktop
alias vi=/usr/local/bin/nvim
alias vim=/usr/local/bin/nvim
```

neovim 配置:

1. [使用nvim配置文件, 将配置文件复制到~/.config目录下](./My_config/nvim) 
2. [在~/.config目录下配置.zshenv文件](./My_config/.zshenv) 


# 大量工具

参考：

- https://github.com/xuxiaodong/awesome-shell/blob/master/README_ZH-CN.md
- https://github.com/bayandin/awesome-awesomeness
## 配置shell

1.首先联网，然后修改apt update的源，
## bashmarks

shell的书签，因为总是cd浪费时间，所以添加书签更方便

启动：

```
source ~/.local/bin/bashmarks.sh
```

编辑：

```
vi ~/.local/bin/bashmarks.sh
```

## tmux

终端复用神器

启动：

```
tmux
```

进入：

```
tmux attach -t 0
```

关闭窗口：

```
control b + &
```

退出但是不关闭：

```
control b + d
```

- https://linuxtoy.org/archives/scripting-tmux.html
- http://mingxinglai.com/cn/2012/09/tmux/

## gogh

shell 主题颜色

使用：鼠标右键进行切换

官网：https://github.com/Mayccoll/Gogh

## surfingkeys

使用：浏览器也可以像vim一样不使用鼠标哦～

链接：https://chrome.google.com/webstore/detail/surfingkeys/gfbliohnnapiefjpjlpjnehglfpaknnc?hl=zh-CN

## ohmyzsh

ohmyzsh 中是配合zsh shell的一个框架，能够大幅度提高你的效率，里面有很多插件可
以是使用。


### vim-like

让你的shell可以像用vim一样。

地址：https://github.com/ohmyzsh/ohmyzsh/tree/master/plugins/vi-mode

使用方式：

编辑~/.zshrc

```
"add vi-mode to plugins"
plugins=(... vi-mode)
```

打开终端，使用i进入插入模式，使用esc或者control + [进入normal模式，进入normal
模式后，就和vim一样了。

- [x] 好像没有配置好，使用的并不流畅，等下看看如何设置

```
vi ~/.zshrc
# 输入下面命令即可
bindkey -v
```

好像默认使用control + [退回到normal模式下，或者直接按esc，并不能进行按键映射。

在normal模式下，按j k 可以调用之前或者之后的命令
[所有可以定义的快捷键](http://bolyai.cs.elte.hu/zsh-manual/zsh_14.html) 


## mac 好用的软件

magnet：窗口管理，让你可以使用快捷键快速管理程序窗口

注意使用的时候不能全屏放大。可以自己设置快捷键。

如何切换这些管理的窗口呢？

好像只能使用command+tab键进行切换。
