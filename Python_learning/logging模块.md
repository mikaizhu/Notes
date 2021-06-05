[TOC]

# 日志模块的使用

**建议使用字典的方式配置：**

```
d = {
    'version':1,
    'formatters':{
        'message':{
            'format':'%(message)s',
        },
        'time_message':{
            'format':'%(asctime)s %(message)s',
            'datefmt':'%Y/%m/%d %H:%M:%S',
        }
    },
    'handlers':{
        'console':{
            'class':'logging.StreamHandler',
            'level':'INFO',
            'formatter':'message'
        },
        'file':{
            'class':'logging.FileHandler',
            'filename':'train.log',
            'level':'INFO',
            'formatter':'time_message'
        },
    },
    'loggers':{
        'logger':{
            'level':'DEBUG',
            'handlers':['file', 'console'],
        },
    },
}
```

**使用方法：**

1. **创建一个tools.py文件，在文件里面写入下面代码：**

```
def get_logging_config(file_name='trian.log'):
    d = {
        'version':1,
        'formatters':{
            'message':{
                'format':'%(message)s',
            },
            'time_message':{
                'format':'%(asctime)s %(message)s',
                'datefmt':'%Y/%m/%d %H:%M:%S',
            }
        },
        'handlers':{
            'console':{
                'class':'logging.StreamHandler',
                'level':'INFO',
                'formatter':'message'
            },
            'file':{
                'class':'logging.FileHandler',
                'filename':file_name,
                'level':'INFO',
                'formatter':'time_message'
            },
        },
        'loggers':{
            'logger':{
                'level':'DEBUG',
                'handlers':['file', 'console'],
            },
        },
    }
    return d
```

2. **在其他py文件中调用这个文件的函数**

```
from tools import get_logging_config
import logging.config


logging.config.dictConfig(get_logging_config(file_name='mian.log'))

logger = logging.getLogger('logger')
```

3. **使用logger代替print**

```
logger.info('this is info')
logger.info('this is info')
logger.debug('this is debug')
logger.error('this is error')
logger.critical('this is critical')
```

==要想修改记录等级，直接在字典中修改即可==

**如果使用函数接口：**

```
import logging

def set_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(asctime)s %(message)s', datefmt='%Y/%d/%m %I:%M:%S')

    # 设置记录文件名和记录的形式
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)

    # 设置让控制台也能输出信息
    file_stream = logging.StreamHandler()
    file_stream.setFormatter(formatter)

    # 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(file_stream)
    return logger


logger = set_logger()
```



```
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)s %(asctime)s %(message)s')

# 设置记录文件名和记录的形式
file_handler = logging.FileHandler('test2.log')
file_handler.setFormatter(formatter)

# 设置让控制台也能输出信息
file_stream = logging.StreamHandler()
file_stream.setFormatter(formatter)

# 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)
logger.addHandler(file_stream)


try:
    assert 1 == 2
except AssertionError:
    logger.error('1 is not equal to 2')

logger.info('this is a info')
logger.debug('this is a debug')
logger.critical('this is a critical')
```

设置成函数的形式：

```
import logging

def set_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s %(asctime)s %(message)s', datefmt='%Y/%d/%m %I:%M:%S')

    # 设置记录文件名和记录的形式
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)

    # 设置让控制台也能输出信息
    file_stream = logging.StreamHandler()
    file_stream.setFormatter(formatter)

    # 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(file_stream)
    return logger


logger = set_logger()

try:
    assert 1 == 2
except AssertionError:
    logger.error('1 is not equal to 2')

logger.info('this is a info')
logger.debug('this is a debug')
logger.critical('this is a critical')
```

## 使用详细介绍

如果想简单使用，直接使用下面代码即可。记得将`print`函数换成`logging.info()`

```
import logging

logging.basicConfig(filename='train.log', level=logging.INFO, filemode='a')
```

`filemode`文件读写模式，default=a，mode=a，不会删除原来文件内容，只是在文件后面不断添加信息

if mode=w，则每次使用都会删除原来的信息。

**显示如下：**

```
INFO:root:0
INFO:root:1
INFO:root:2
INFO:root:3
INFO:root:4
```

**可以改变输出模式：**

```
import logging

logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    filemode='a',
    format='%(message)s'
)
```

```
# 直接输出如下
0
1
2
3
4
```

添加运行时间：

```
import logging

logging.basicConfig(
    filename='train.log',
    level=logging.INFO,
    filemode='a',
    format='%(asctime)s %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S'
)
```

```
2020/12/02 03:11:14 0
2020/12/02 03:11:14 1
2020/12/02 03:11:14 2
2020/12/02 03:11:14 3
```

[显示的时间格式和time模块一样，注意有大小写分别](https://docs.python.org/zh-cn/3/library/time.html#time.strftime)

**其实挺好记的：**

```
Y：year
m：month
d：day
H：hour
M：minute
S：second
```

## 高级用法

**首先，我们要有一个观念。**

- 记录器
- 处理器
- 输出格式
- 过滤器

这些观念，都和现实生活很像

- 定义一个记录器，就像买了几支笔。
- 定义一个处理器，告诉笔要写在哪里
- 定义输出格式，告诉要怎么记录
- 定义过滤器，就是过滤一些信息

**下面来编程实现：**

```
import logging

# 设置记录器，并给记录器取名字，每个记录器就像一支笔
logger = logging.getLogger('zwl')

# 设置处理器，每个处理器就是方便管理这些笔，将信息输入到合适的位置
# 告诉笔往哪边输出信息
consolHandle = logging.StreamHandler()

fileHandle = logging.FileHandler(filename='t.log') # 这里肯定要传入文件名字

# 设置输出格式
formater = logging.Formatter('%(message)s')
```

**其中过滤器不是必须的，可以不用设置**

**怎么将这些串联起来呢？**

1. 给处理器添加输出格式

```
consolHandle.setFormatter(formater)
fileHandle.setFormatter(formater)
```

2. 给记录器添加处理器

```
logger.addHandler(consolHandle)
logger.addHandler(fileHandle)
```

**完整代码如下：**

```
import logging

# 设置记录器，并给记录器取名字，每个记录器就像一支笔
logger = logging.getLogger('zwl')
# logger.setLevel(logging.INFO)

# 设置处理器，每个处理器就是方便管理这些笔，将信息输入到合适的位置
# 告诉笔往哪边输出信息
consolHandle = logging.StreamHandler()
consolHandle.setLevel(logging.DEBUG)

fileHandle = logging.FileHandler(filename='t.log')
fileHandle.setLevel(logging.INFO)

# 设置输出格式
formater = logging.Formatter('%(message)s')

consolHandle.setFormatter(formater)
fileHandle.setFormatter(formater)


logger.addHandler(consolHandle)
logger.addHandler(fileHandle)

logger.info('this is info')
logger.debug('this is debug')
logger.critical('this is critical')
```

但是，这样控制台输出和文件写入的，只有critical信息，why？

**我们还应该给比添加level**

**如果不给笔添加level，默认是critical信息，只有以上才会记录，如果控制器想输入info信息，但笔压根就不写，那也没办法。所以要设置记录器的level**

```
import logging

# 设置记录器，并给记录器取名字，每个记录器就像一支笔
logger = logging.getLogger('zwl')
# logger.setLevel(logging.INFO)
```

## 使用字典配置日志

**基础设置：字典的键和值，如下形式**

```
d = {
    'version':1,
    'formatters':{},
    'filters':{}, # 可选
    'handlers':{},
    'loggers':{},
}
```

**然后可以向里面添加内容**

补全代码后如下：

```
import logging
import logging.config

d = {
    'version':1,
    'formatters':{
        'simple':{
            'format':'%(message)s',
            'datefmt':'%Y%m%d %H%M%S',
        },
    },
    'handlers':{
        'console':{
            'class':'logging.StreamHandler',
            'level':'INFO',
            'formatter':'simple'
        },
        'file':{
            'class':'logging.FileHandler',
            'filename':'train.log',
            'level':'INFO',
            'formatter':'simple'
        },
    },
    'loggers':{
        'simpleExample':{
            'level':'INFO',
            'handlers':['file', 'console'],
        },
    },
    'root':{
        'level':'INFO',
    }
}

logging.config.dictConfig(d)
logger = logging.getLogger('simpleExample')

logger.info('this is info')
logger.debug('this is debug')
logger.error('this is error')
logger.critical('this is critical')
```

**尝试对里面的内容进行修改(更大胆的尝试)：**

```
import logging
import logging.config


d = {
    'version':1,
    'formatters':{
        'simple1':{
            'format':'%(message)s',
        },
        'simple2':{
            'format':'%(asctime)s %(message)s',
            'datefmt':'%Y%m%d %H%M%S',
        }
    },
    'handlers':{
        'console':{
            'class':'logging.StreamHandler',
            'level':'INFO',
            'formatter':'simple2'
        },
        'file':{
            'class':'logging.FileHandler',
            'filename':'train.log',
            'level':'INFO',
            'formatter':'simple1'
        },
    },
    'loggers':{
        'simpleExample':{
            'level':'INFO',
            'handlers':['file', 'console'],
        },
        'logger1':{
            'level':'DEBUG',
            'handlers':['file'],
        },
        'logger2':{
            'level':'ERROR',
            'handlers':['console'],
        },
    },
    'root':{
        'level':'INFO',
    }
}

logging.config.dictConfig(d)
logger = logging.getLogger('logger2')

logger.info('this is info')
logger.debug('this is debug')
logger.error('this is error')
logger.critical('this is critical')
```

## 讲解使用字典配置

首先，我们要知道日志模块重要的组成部分：

- formatters(输出格式)
- handlers(处理器)
- loggers(记录器)
- filters(过滤器)

**所以字典中通常包含以下部分：**

```
d = {
    'version':1,
    'formatters':{},
    'filters':{}, # 可选
    'handlers':{},
    'loggers':{},
}
```

**然后介绍里面的参数：**

### formatters参数

**输出的格式**

```
formatters
	-- format_name - {}
```

Eg:

```
'formatters':{
        'simple1':{
            'format':'%(message)s',
        },
        'simple2':{
            'format':'%(asctime)s %(message)s',
            'datefmt':'%Y%m%d %H%M%S',
        }
    },
```

1. formatters的值必须为字典
2. 字典的键是格式名
3. 可以设置多种格式
   - 格式主要含有两种参数
   - format
   - datefmt

### handlers参数

指定输出到哪里，主要有控制台和日志文件

```
handlers
	-- handlers_name - {}
```

Eg:

```
'handlers':{
        'console':{
            'class':'logging.StreamHandler',
            'level':'DEBUG',
            'formatter':'simple2'
        },
        'file':{
            'class':'logging.FileHandler',
            'filename':'train.log',
            'level':'DEBUG',
            'formatter':'simple1'
        },
    },
```

1. handlers传入的值，必须是字典
2. 传入的字典的键，是处理器的名字，可以设置多个处理器，不一定是console和file，这样只是好记
3. 处理器中主要含有四种参数
   - class 必须传入的参数，类型如上，必须对应
   - level，处理器的等级
   - formatter，设置处理器的输出格式
   - filename，如果输出到日志中，才有这个参数

### loggers参数

相当于笔，记录器，可以随意选择使用哪个记录器

```
loggers
	-- loggersname - {}
```

Eg:

```
'loggers':{
        'simpleExample':{
            'level':'INFO',
            'handlers':['file', 'console'],
        },
        'logger1':{
            'level':'INFO',
            'handlers':['file'],
        },
        'logger2':{
            'level':'ERROR',
            'handlers':['console'],
        },
```

1. loggers参数传入的必须是字典，可以设置多个记录器
2. 记录器中主要含有2个参数
   - level，记录器的等级
   - handlers，为记录器配置控制器，传入的是列表的形式，可以传入多个控制器

==注意：如果记录器的等级要高于控制器的等级，那么只会按记录器的等级输出==

### 记录错误

如果直接使用logger.error()记录错误信息，不会记录错误出现在哪一行

**可以使用下面方式：**

```
try:
	pass
except Exception as e:
	logger.excption(e)
```



# 参考教程

1. bilibili

https://www.bilibili.com/video/BV1sK4y1x7e1?from=search&seid=3502857864397522115

2. zhihu

https://zhuanlan.zhihu.com/p/86284675

这里介绍了yml的格式，其实和字典很像，可以参考
