<!--ts-->
* [介绍](#介绍)
* [画图](#画图)
   * [散点图绘制](#散点图绘制)
   * [绘制线段](#绘制线段)
   * [条形图绘制](#条形图绘制)
   * [直方图的绘制](#直方图的绘制)
   * [plotly.express 说明](#plotlyexpress-说明)
   * [3D图](#3d图)

<!-- Added by: mikizhu, at: 2021年 6月 8日 星期二 21时32分30秒 CST -->

<!--te-->
# 介绍

Plotly是一个python画图模块，可以优美画图

参考官网：https://github.com/plotly/plotly.py/tree/master/doc/python

参考视频：https://www.bilibili.com/video/BV13E411Z7XH?from=search&seid=9365475051760847749

# 画图

## 散点图绘制

```
import pandas as pd
# plotly 一般使用这个模块来进行画图
import plotly.graph_objects as go

# 首先绘制线条, mode='markers' 说明使用散点，不要绘制为线段
line1 = go.scatter(x=data['date'], y=data['auckland'], mode='markers')
# 然后将线条放到图画上
fig = go.Figure(line1)
fig.update_layout(
  title='New Zealand Weather',
  xaxis_title='Date',
  yaxis_title='Weather',
)
# 让图片显示出来
fig.show()
```

绘制不同颜色

```
line1 = go.scatter(
  x=data['date'],
  y=data['auckland'],
  mode='markers',
  marker={'color':data['color']}, # 这里使用marker参数，注意data应该传入0，1，
  2类别
)
# 然后将线条放到图画上
fig = go.Figure(line1)
```


## 绘制线段

```
import pandas as pd
# plotly 一般使用这个模块来进行画图
import plotly.graph_objects as go

# 首先绘制线条
line1 = go.Scatter(x=data['DATE'], y=data['Auckland'])
line2 = go.Scatter(x=data['DATE'], y=data['Auckland'])
# 然后将线条放到图画上
fig = go.Figure(line1, line2)
fig.update_layout(
  title='New Zealand Weather',
  xaxis_title='Date',
  yaxis_title='Weather',
)
# 让图片显示出来
fig.show()
```
## 条形图绘制

```
bar1 = go.Bar(
  x=data["date"],
  y=data['auckland'],
  text=data['auckland'], # 使得条形图可以显示数据
  textposition='outline', # 选择条形图的数据是显示在里面还是条形图上边
)
bar2 = go.Bar(
  x=data["date"],
  y=data['auckland'],
  text=data['auckland'], # 使得条形图可以显示数据
  textposition='outline', # 选择条形图的数据是显示在里面还是条形图上边
)
# 可以将两个条形图画在上面
fig = go.Figure([bar1, bar2])
fig.show()
```

## 直方图的绘制

```
hist = go.Histogram(
  x=data['Auckland'],
  xbins={'size':10}, # 设置直方图的每个区间为10
)
fig = go.Figure(hist)
fig.update_layout(bargap=0.1) # 设置直方图的有一些间隙为0.1
fig.show()
```

## plotly.express 说明

这个比之前的go模块要好用很多

```
import plotly.express as px

fig = px.scatter(
  data, # data 是一个dataframe，即表格数据
  x='Length',
  y='Width',
  color='Name' # 会自动根据name来进行分类颜色
)

fig.show()
```

## 3D图

```
line = go.Scatter3d(
  x=data['x'],
  y=data['y'],
  z=data['z'],
  mode='markers', # 设置只画点
  marker={'size':5, color='red'} # 设置点的大小
)
fig = go.Figure(line)
fig.show()
```

也可以使用px进行绘制3d图，更加方便

```
fig = px.scatter_3d(
  data, # data是一个data frame
  x='x', # x 是column的名字
  y='y',
  z='z',
  color='color', # 这样会给每个类别画不同颜色
)
fig.show()
```

之前的plotly有在线版和离线版，现在统一了

```
import plotly.offline as of                    #保存图表，相当于plotly.plotly as py，同时增加了离线功能
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
