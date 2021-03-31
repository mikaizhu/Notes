## 时间设置

假设现在有个时间序列

```
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots()

# 可以设置的参数MONTHLY, WEEKLY, DAILY
rule = mdates.rrulewrapper(mdates.DAILY, interval=1) # 设置时间刻度, interval为时间间隔
loc = mdates.RRuleLocator(rule) # 对时间刻度进行封装

dateFmt = mdates.DateFormatter('%m/%d') 

ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(dateFmt)

ax.tick_params(axis='both', direction='out', labelsize=10)

date1 = datetime.date(2021, 3, 31)
date2 = datetime.date(2021, 3, 25)
delta = datetime.timedelta(days=1)
dates = mdates.drange(date2, date1, delta)

ax.plot_date(dates, week_users, 'b-', alpha=0.7)
fig.autofmt_xdate()
plt.show()
```

> <img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gp31gvam9kj30le0diwfv.jpg" alt="image.png" style="zoom:50%;" />
>
> 
>
> 请参考 matplotlib 精进这本书第42页
>
> 补充：如果要日期显示为英文
>
> ```
> dateFmt = mdates.DateFormatter('%b/%d') # %b表示英文的月份
> ```
>
> 日期格式参考：https://www.pythonf.cn/read/51756

