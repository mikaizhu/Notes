import random
import numpy as np
import torch
from pathlib import Path
import logging
import logging.config

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_logging_config(file_name='trian.log'):
    # 默认创建log文件夹，存放所有日志文件
    path = Path('./log')
    if not path.exists():
        path.mkdir()
    else:
        # 删除日志原有的内容
        path = path/Path(file_name)
        with path.open('w') as f:
            f.truncate()

    file_name = str(path)

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
                'level':'INFO',
                'handlers':['file', 'console'],
            },
        },
    }

    return d

if __name__ == '__main__':
    set_seed(42)
    logging.config.dictConfig(get_logging_config(file_name='train.log'))
    logger = logging.getLogger('logger')
    logger.info('123')

