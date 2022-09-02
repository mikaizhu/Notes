在Linux中有bash_profile，然后linux开机会自动执行里面的代码。

```
vi ~/.bash_profile
```

在mac中，里面也有类似的文件

```
vi ~/.zshenv
```

在里面写入想要开机自动执行的代码即可

```
#!/bin/bash

source ~/.bash_profile
```

配合tmux使用，就非常方便了