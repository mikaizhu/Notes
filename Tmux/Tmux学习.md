# tmux按键映射

为了和vim相对应，所以将按键映射成vim的形式

```
# 将前缀按键设置成control + x
set -g prefix C-x

# 取消默认的绑定
unbind C-b

# 将窗格切换按键进行映射
bind-key k select-pane -U
bind-key j select-pane -D
bind-key h select-pane -L
bind-key l select-pane -R
```

参考网址：http://mingxinglai.com/cn/2012/09/tmux/

# tmux概念

**重要概念**

- 首先我们打开tmux，就是创建了一个session
- 然后在session里面，可以创建很多个工作windows
- 然后在工作windows里面，可以分成很多个pane。利用每个pane完成任务

![image.png](http://ww1.sinaimg.cn/large/005KJzqrly1gjy345c72gj30dv093abs.jpg)

# tmux按键
- 创建session 后面是名称

```
tmux new -s py
```

- 分屏操作

```
control x % 竖分屏
control x " 横分屏
```

- 新建Windows 也就是在tmux的session中新建Windows

```
control x c
```

- 切换窗口

```
control x n 下一个窗口
control x p 上一个窗口
```

- 查看有多少session

```
tmux ls
```

- 切换到session

```
tmux attach -t session name
```

# 窗口和session操作

- 退出session和Windows，同时关闭后台程序

```
exit
```

- 只退出窗口，不关闭后台

```
control x d
```

- 切换session，在任意一个session中即可

```
control x s # 先列出所有会话
jk进行上下选择
```

- 列出所有session

```
control x s
```

- 给session重新命名

```
control x $

tmux rename-session -t 0 <new-name>
```

