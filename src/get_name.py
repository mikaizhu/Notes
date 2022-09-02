#!/Users/zwl/miniconda3/bin/python3
from pathlib import Path as p

# 有些笔记下有doc目录，为了将这些文件名，存入readme中方便查看
path = p('/Users/zwl/Desktop/Notes/Tools/Git/doc')
root = p('./doc')
res = list(path.glob('*.md'))
file_name_list = list(map(lambda x: str(x.name), res))
file_path_list = list(map(str, [root/i.name for i in res]))

for i in range(len(res)):
    print(f'[{file_name_list[i]}]({file_path_list[i]})', end='\n')
    print()

