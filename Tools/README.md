## 大量工具

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
