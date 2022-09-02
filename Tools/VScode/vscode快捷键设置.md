# 设置快捷键

今天弄了我一个下午的pycharm远程调试，发现vscode比pycharm好用多了，一下就调好了远程ssh，而且vscode是免费的，pycharm真菜

**注意，要设置什么按键，只要去vscode官方查看下就好了**

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

在文件中输入下面代码，注意是把下面代码添加进去

```
{   
    // Controls if quick suggestions should show up while typing
    "editor.quickSuggestions": {
        "other": true,
        "comments": false,
        "strings": false
    },

        // Controls whether suggestions should be accepted on commit characters. For example, in JavaScript, the semi-colon (`;`) can be a commit character that accepts a suggestion and types that character.
    "editor.acceptSuggestionOnCommitCharacter": true,

    "editor.tabCompletion": true,

    // Controls if suggestions should be accepted on 'Enter' - in addition to 'Tab'. Helps to avoid ambiguity between inserting new lines or accepting suggestions. The value 'smart' means only accept a suggestion with Enter when it makes a textual change
    "editor.acceptSuggestionOnEnter": "on",
    // Controls the delay in ms after which quick suggestions will show up.
    "editor.quickSuggestionsDelay": 10,

    // Controls if suggestions should automatically show up when typing trigger characters
    "editor.suggestOnTriggerCharacters": false,

    // Controls if pressing tab inserts the best suggestion and if tab cycles through other suggestions
    "editor.tabCompletion": "on",

    // Controls whether sorting favours words that appear close to the cursor
    "editor.suggest.localityBonus": true,

    // Controls how suggestions are pre-selected when showing the suggest list
    "editor.suggestSelection": "recentlyUsed",

    // Enable word based suggestions
    "editor.wordBasedSuggestions": false,

    // Enable parameter hints
    "editor.parameterHints.enabled": false,
    "code-runner.saveAllFilesBeforeRun": true,
    "code-runner.saveFileBeforeRun": true,
    "python.defaultInterpreterPath": "/Users/mikizhu/miniconda3/envs/py38_env/bin/python3.8",
    "terminal.integrated.inheritEnv": false,
    "editor.parameterHints": false,
    "typescript.tsdk": "",
    "editor.quickSuggestions": false,
    "workbench.colorTheme": "Monokai",
    "explorer.confirmDragAndDrop": false,
    // "emmet.triggerExpansionOnTab": true,
    "python.jediEnabled": true,

    //debug 配置
    "launch": {
    
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${cwd}"
        }
        
    },
    "python.languageServer": "Jedi",
    "jupyter.sendSelectionToInteractiveWindow": false,
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

