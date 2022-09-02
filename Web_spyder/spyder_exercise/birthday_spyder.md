# 介绍农历生日的爬虫

> 爬取的网站：https://www.sojson.com/time/gongli.html
>
> 参考到的教程：https://blog.csdn.net/huilan_same/article/details/52246012 解决网站中的选择框如何用selenium 爬取



## **主要用到的知识：**

- selenium模块
- re模块

## **分析网站：**

**按f12不行，于是使用鼠标右键审查元素，查看框图如下：**

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gnbal9pa1ej31wc0iywl8.jpg)

**发现上面的框是由select 和option组成，那么用selenium如何自动化控制这个选择框呢？**

```
from selenium.webdriver.support.ui import Select # 使用这个模块
```

**于是，整个爬虫代码如下：**

```
from selenium import webdriver
from selenium.webdriver.support.ui import Select
import time
import pickle

'''
参考教程：https://blog.csdn.net/huilan_same/article/details/52246012

爬取网站：https://www.sojson.com/time/gongli.html
'''

year = [str(1997 + i) for i in range(100)] # 生成要爬取的所有年份
ans = [] # 保存爬取的结果，农历转公历

driver = webdriver.Chrome()

path = r'https://www.sojson.com/time/gongli.html' # 这里填写url
driver.get(path) # 启动浏览器并打开网页
time.sleep(3)

for each_year in year: # 使用循环开始抓取所有年份
    print('dealing {}'.format(each_year))
    time.sleep(1)
    s1 = Select(driver.find_element_by_id('YYear'))  # 实例化Select，找到选择框
    s1.select_by_value(each_year)  # 选择value="==="的项

    time.sleep(1)
    s2 = Select(driver.find_element_by_id('YMonth'))
    s2.select_by_value('10')

    time.sleep(1)
    s3 = Select(driver.find_element_by_id('YDay'))
    s3.select_by_value('25')

    click = driver.find_element_by_css_selector('#yin > form > button')
    click.click()

    time.sleep(1)

    result = driver.find_element_by_id('result')

    ans.append(result.text) # 保存查询的结果

driver.quit() # 退出浏览器

# 数据保存
pickle.dump(ans, open('birthday.pkl', 'wb'))
pickle.dump(year, open('year.pkl', 'wb'))
```

## 数据处理

上面爬取的数据不美观，所以使用re模块

```
import pickle
import pandas as pd
import re


birth = pickle.load(open('birthday.pkl', 'rb'))
year = pickle.load(open('year.pkl', 'rb'))

for idx, i in enumerate(birth):
    birth[idx] = re.findall('[0-9]+年[0-9]+月[0-9]+日.*星期.', birth[idx])[0] # re 模块清洗数据

df = list(zip(year, birth))
df = pd.DataFrame(df)

df.columns = ['year', 'birth']

df = df.set_index(df['year'])
del df['year']

df.to_csv('birthday.csv', sep='\t')
```



**完成啦～**

