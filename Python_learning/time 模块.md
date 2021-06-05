- 时间戳
- 结构化时间对象
- 格式化时间字符串

```
import time
```

## 时间戳

1971年1. 1日0点到现在的秒数

```
time.time() # 生成当前时间的时间戳
```

## 结构化时间对象

```
st = time.loacaltime() # 生成实例化对象
```

```
type(st) # time.struct_time
```

直接输出`st`则

```
time.struct_time(tm_year=2021, tm_mon=1, tm_mday=23, tm_hour=13, tm_min=57, tm_sec=23, tm_wday=5, tm_yday=23, tm_isdst=0)
```

- Tm wday 表示一周的第几天，0表示星期1
- tm yday 表示一年的第几天

两种方法访问st的内部信息

```
print('time is {}-{}-{}'.format(st[0], st[1], st[2]))
print('time is {}'.formate(st.tmwday + 1))
```

## 格式化的时间字符串

```
time.ctime() # 'Sat Jan 23 14:02:39 2021'
```

**更一般化的结构**

```
time.strftime('%Y-%m-%d %H:%M:%S') # '2021-01-23 14:04:09'
time.strftime('%Y年%m月%d日 %H时%M分%S秒') # '2021-01-23 14:04:09'
```

## 其他代码

```
time.sleep(2.3)
```

## 三种格式的转换

1. **时间戳转结构化对象**

```
# 获得UTC时间
time.gmtime()
time.gmtime(time.time() - 3600)
```

```
# local time
time.localtime()
time.localtime(time.time() - 3600) # time.struct_time(tm_year=2021, tm_mon=1, tm_mday=23, tm_hour=13, tm_min=16, tm_sec=46, tm_wday=5, tm_yday=23, tm_isdst=0)
```

2. **结构化对象转格式化时间字符串**

```
time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()) # '2021-01-23 14:20:44'
```

3. **格式化字符串转结构化时间对象**

```
strtime = '2021-01-23 14:20:44'
time.strptime(strtime, '%Y-%m-%d %H:%M:%S') # time.struct_time(tm_year=2021, tm_mon=1, tm_mday=23, tm_hour=14, tm_min=20, tm_sec=44, tm_wday=5, tm_yday=23, tm_isdst=-1)
```

