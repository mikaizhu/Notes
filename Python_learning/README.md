# 参考资料推荐

- https://pyzh.readthedocs.io/en/latest/
- python魔法方法指南：https://pyzh.readthedocs.io/en/latest/python-magic-methods-guide.html

# python 日志模块使用

如果想和Linux脚本使用终端输出，使用print不行，只能用logger文件，如果想简单使用
logger，使用下面代码即可

```
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(message)s', level=logging.INFO)
```

or

```
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
```

然后将print修改成logger.info:

```
:%s/print/logger.info
```

