# ssh的设置

平时我们在使用ssh登陆远程服务器的时候，我们一般像下面一样登陆

```
ssh zwl@12345
```

通常后面的ip很难记忆，因此我们可以进行映射

```
vi ~/.ssh/config
```

在里面写入：

```
Host zwl
	HostName 192.169.0.200
	User zwl
	Port 22
```

然后我们就可以像下面一样登陆服务器了,输入密码即可

```
ssh zwl
```

如果要传输文件：

```
ssh localfile 777:Desktop
```

