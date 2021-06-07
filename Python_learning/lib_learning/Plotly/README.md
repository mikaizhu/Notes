# 介绍

Plotly是一个python画图模块，可以优美画图

参考官网：https://github.com/plotly/plotly.py/tree/master/doc/python

# 画图

```
import plotly.offline as py                    #保存图表，相当于plotly.plotly as py，同时增加了离线功能
py.init_notebook_mode(connected=True)          #离线绘图时，需要额外进行初始化
import plotly.graph_objs as go                 #创建各类图表
import plotly.figure_factory as ff             #创建table

of.offline.init_notebook_mode(connected=True)
t0 = go.Scatter(
    y=np.array(train_acc), # train_acc 是一个列表
    x=np.array(range(len(train_acc))),
    mode='lines + markers',
    name='train'
)
t1 = go.Scatter(
    y=np.array(test_acc),
    x=np.array(range(len(test_acc))),
    mode='lines + markers',
    name='test'
)
layout = dict(title='Accuracy', xaxis=dict(title='Epochs'), yaxis=dict(title='acc'))
data = dict(data=[t0, t1], layout=layout)
py.iplot(data, filename='dnn_train') # filename 是保存的图片名字
```
