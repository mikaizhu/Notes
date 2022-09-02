# 参考教程

- https://tianchi.aliyun.com/course/324/3654

# 基础知识

散点图绘制：

```
map_color = {0:'r', 1:'g'}
map_marker = {0:'.', 1:'.'}
# 将字符串转换成数字
color = df.iloc[:, 2].apply(lambda x:label_dict.index(x))
# 将数字类别转换成不同颜色
diff_color = color.apply(lambda x: map_color[x])
#diff_marker = list(color.apply(lambda x: map_marker[x]))
#如果想要不同形状：https://blog.csdn.net/u014571489/article/details/102667570
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=diff_color, marker='.');
```

绘制条形图：

```
x = df.iloc[:, 2].value_counts().index
y = df.iloc[:, 2].value_counts().values
```

## jupyter中matplotlib的含义

参考：https://www.cnblogs.com/emanlee/p/12358088.html

总结：%matplotlib inline 可以在Ipython编译器里直接使用，功能是可以内嵌绘图，并且可以省略掉plt.show()这一步。

# 在折线中标记某些点

```
plt.plot(df)
plt.plot(df, markevery=mark, marker='o')
```

折线图的绘制，参考代码：

```
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

root = 'acc_data'
# 找到所有符合条件的路径
# [[file1, file2, file3, file4], [file1, file2, file3, file4] ...]
all_list = []
for file in list(Path(root).iterdir()):
    for sub1_file in file.iterdir():
        if '.DS_Store' in str(sub1_file):
            sub1_file.unlink()
        sub_list = []
        for sub2_file in sub1_file.iterdir():
            if '.DS_Store' in str(sub2_file):
                sub2_file.unlink()
            temp = []
            for sub3_file in sub2_file.iterdir():
                temp.append(sub3_file.name)
            if 'val_acc.npy' in temp:
                idx = temp.index('val_acc.npy')
                res = list(map(str, list(sub2_file.iterdir())))[idx]
                sub_list.append(res)
            else:
                idx = temp.index('Acc.npy')
                res = list(map(str, list(sub2_file.iterdir())))[idx]
                sub_list.append(res)
        all_list.append(sub_list)

# 将file1， file2， file3， file4 中的文件绘制在一张图片上
# 并保存文件
for i in range(len(all_list)):
    x1, x2, x3, x4 = np.load(all_list[i][0]), np.load(all_list[i][1]), np.load(all_list[i][2]), np.load(all_list[i][3])
    label1 = all_list[i][0].split('/')[-2]
    label2 = all_list[i][1].split('/')[-2]
    label3 = all_list[i][2].split('/')[-2]
    label4 = all_list[i][3].split('/')[-2]
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(x1, 'r,-', label=label1)
    plt.plot(x2, 'gv-', label=label2)
    plt.plot(x3, 'b.-', label=label3)
    plt.plot(x4, 'c*-', label=label4)
    title = '/'.join(all_list[i][1].split('/')[1:3])
    save_path = '/'.join(all_list[i][0].split('/')[:3])
    file_name = all_list[i][0].split('/')[2] + '.png'
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('/'.join([save_path, file_name]))
    plt.show()
```

