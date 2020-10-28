# 设置快捷键

今天弄了我一个下午的pycharm远程调试，发现vscode比pycharm好用多了，一下就调好了远程ssh，而且vscode是免费的，pycharm真菜。

## 取消提示

默认下载的vscode，会有代码提示这些，函数功能也会自动提示，对编程很不友好。所以，先进行以下设置

先看看下面这个快捷键：

```
command + shift + p
```

这个按键可以调出搜索框出来，主要是对vscode的文件进行设置和搜索。

**设置取消提示，这里会取消两个提示：**

- 取消print类似的输入提示
- 取消print函数内部的功能提示

首先输入：

```
command + shift + p
```

然后在框中查找文件：

```
settings.json
```

在文件中输入下面代码，重启vscode即可

```
{
    "python.dataScience.sendSelectionToInteractiveWindow": false,
    "code-runner.saveAllFilesBeforeRun": true,
    "code-runner.saveFileBeforeRun": true,
    "python.defaultInterpreterPath": "/home/zwl/miniconda3/envs/python38_env/bin/python",
    "workbench.colorTheme": "Monokai",
    "terminal.integrated.inheritEnv": false,
    "editor.parameterHints": false
}
```

## 按键设置

**上面取消了输入提示，但是我希望，输入tab还是可以自动补全，输入特定按键，还是可以查看有什么输入**

```
command + shift + p
```

在搜索框中查找文件

```
keyboard shortcuts
```

在快捷键搜索框中输入：

```
suggest
```

然后将快捷键设置为tab即可

如果要提示参数,可以搜索下面关键字

```
parameter
```

会提示快捷键是,space 也就是空格

```
command + shift + space
```

目前设置结束

