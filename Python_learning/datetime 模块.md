**Datetime主要的四个类对象：**

- datetime.date
- datetime.time
- datetime.datetime
- datetime.timedelta

## datetime.date

1. **生成日期**

```
datetime.date.today() # 生成今天的时间
```

```
datetime.date(2020, 8, 25) # 生成日期的一般方法
```

```
datetime.date.fromtimestamp(time.time()) # 根据时间戳生成日期 datetime.date(2021, 1, 23)
```

2. **获得实例对象的属性**

```
d = datetime.date.fromtimestamp(time.time()) # 根据时间戳生成日期
d.year
d.month
d.day
```

3. **将date time 对象转换成结构化时间对象**

```
d # datetime.date(2021, 1, 23) 实例对象
```

```
d.timetuple() # 调用实例对象的方法 time.struct_time(tm_year=2021, tm_mon=1, tm_mday=23, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=5, tm_yday=23, tm_isdst=-1)
```

4. **修改时间**

```
d.replace(2022,2 ,24)
d.replace(day=24) # 指定时间参数修改
```

5. 常用函数

```
d.weekday()
d.strftime('%Y年%m月%d日')
```

## datetime.time

```
t = datetime.tiem(15, 10, 45, 8888) # 生成时间，小时，分，秒，微秒
```

```
t.hour
t.minute
t.second
```

## datetime.datetime

```
dt = datetime.datetime(2020, 8, 20, 13, 22, 34, 8888) # 生成时间
```

```
datetime.datetime.now() # datetime.datetime(2021, 1, 23, 14, 53, 5, 489866)
datetime.datetime.today() # datetime.datetime(2021, 1, 23, 14, 53, 25, 67999)
datetime.datetime.fromtimestamp(time.time()) # datetime.datetime(2021, 1, 23, 14, 53, 54, 115123)
```

**将字符串转换成dt的类型**

```
dt = datetime.datetime.strptime('2020-7-15 3:13:45', '%Y-%m-%d %H:%M:%S') # datetime.datetime(2020, 7, 15, 3, 13, 45)
```

注意前面传入的必须是字符串

**一些属性**

```
dt.year
dt.month
dt.day
dt.hour
dt.second
```

## 格式转换方法

1. **dt转结构化对象**

```
dt.timetuple()
```

2. dt 转时间戳

```
dt.timestamp()
```

3. dt转格式化字符串

```
dt.strftime('%Y-%m-%d %H-%M-%S')
```

4. 时间戳转dt对象

```
datetime.datetime.fromtimestamp(time.time())
```

5. 格式化对象转dt

```
datetime.datetime.strptime('2020-7-14 3:13:46', '%Y-%m-%d %H-%M-%S')
```

6. 结构化对象转dt

```
datetime.datetime.fromtimestamp(time.mktime(time.localtime()))
```

## 时间运算 datetime.timedelta

**1. 生成时间差**

```
datetime.timedelta(days=10)
datetime.timedelta(days=10, hours=5)
datetime.timedelta(days=-10)
datetime.timedelta(hours=75)
datetime.timedelta(weeks=2)
```

```
datetime.timedelta(days=10)
datetime.timedelta(days=10, seconds=18000)
datetime.timedelta(days=-10)
datetime.timedelta(days=3, seconds=10800)
datetime.timedelta(days=14)
```

2. **时间计算**

```
dt = datetime.datetime.today() # datetime.datetime(2021, 1, 23, 15, 20, 10, 722682)
delta = datetime.timedelta(days=10)
target = dt + delta # datetime.datetime(2021, 2, 2, 15, 20, 10, 722682)
```

**注意只能dt对象时间计算**

```
dt1 = datetime.datetime.today()
dt2 = datetime.datetime.utcnow()
dt2 - dt1
datetime.timedelta(days=-1, seconds=57620, microseconds=818230)
```

