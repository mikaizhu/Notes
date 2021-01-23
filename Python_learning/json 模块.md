## json基础

**什么是json？**

json又叫java script object notation -- **java script 对象表示法**

JSON 是轻量级的文本数据交换格式，是一种独立语言，不同于任何语言

JSON 比XML更小，更快，更容易解析

**JSON长什么样子？**

```
{"firtstname":"John", "lastname":"Doe"}
```

和python中的字典长的很像，但是本质上有区别

## 格式转换问题

**python和json格式的转换，在数据类型上会有变化**

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1gmxoqzhal6j31mw12eqjo.jpg" alt="image.png" style="zoom: 33%;" />

## json的使用方法

```
json.dump(obj, fp) # 将python类型数据转换并保存到json格式的文件内，读取文件
json.dumps(obj) # 将python数据类型转换成json格式的字符串
json.load(fp) # 从json格式的文件中读取数据并转换成python类型
json.loads(s) # 将json格式的字符串转换成python类型
```

