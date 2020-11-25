# 日志模块的使用

直接复制下面即可

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
    file_handler.setLevel(logging.ERROR)
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

