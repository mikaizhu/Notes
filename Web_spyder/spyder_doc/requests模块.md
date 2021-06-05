[TOC]

## requests模块使用

### **使用方法**

- 首先我们要注意请求网页是用的post请求还是get请求，这里在网页控制台里面可以看到

```
import requests

url = 'www.baidu.com'
responce = requests.get(url) # 使用get请求
text = responce.text
```

**查看一些信息**

- **打开控制台**
- **点击network选项**
- **然后点击文件，查看信息**
- **如果没有出现文件，记得刷新网页！！！**

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goypgucjqhj324a0smguh.jpg)

**常见的反爬机制**

1. **user agent请求头**

```
user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36
```

有时候网页会检查发送的请求头，如果不是常见的浏览器，则可能服务器不会理会你的请求。所以在request上应该设置UA伪装。

**使用requests的header参数即可**

```
headers = {
'User-Agent':Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36
}
requests.get(url, headers=headers)
```

**课程1的代码：**

```
import requests

url = r'https://www.google.com.hk/search' # 注意谷歌的搜索地址是这个，只有用requests访问这个网址才能搜索到东西

param = {
'q':'requests模块使用',
'client': 'gws-wiz',
'hl': 'zh-CN',
'pq': 'requests模块使用'
}

header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.get(url=url, headers=header, params=param) # 发送请求

# 爬取的数据保存
with open('./spy.html', 'w') as f:
    f.write(res.text)
```

**其中的小细节：**

1. **怎么知道访问这个网址就好了呢？**

如果我只是requests请求这个网页 https://www.google.com.hk/，结果得不到正确的答案

好像目前只能通过浏览器搜索框来观察，前面是域名，问号后面对应的是请求的参数

![](http://ww1.sinaimg.cn/large/005KJzqrgy1goyorf1pw6j30x602gaa9.jpg)

> 我们知道了浏览器的url是可以带参数的，requests就是访问url，requests模块的第二个参数就是params，url中的参数都可以封装到params里面。所以requests请求的url到search就好了，后面的参数可以不要，封装到params字典中

2. **params参数怎么设置呢？**

点击搜索后，刷新网页，可以看到服务器会发送几个文件，点击文件进去，可以看到请求头，这个就是params参数

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyohdiac0j325g17enfy.jpg)

3. **什么是XHR数据包呢？点击浏览器里面的XHR就能看到？**

比如我们在搜索的时候，并不是每次搜索到的界面都是一样的，所以我们看到的网页都是动态的，什么是静态网页呢？比如我写的一篇博客，每次访问都是一样的，这就是静态的。XHR和ajax有很大关系，让我们搜索的界面是动态的，即我们可以动态和服务器进行数据交换，所以一般搜索的时候都是观察XHR，很多XHR里面的文件，就是我们搜索界面呈现的内容。

4. **get请求和post请求有什么区别？**

参考：https://www.zhihu.com/question/28586791

总的来说，get请求就是访问资源，比如谷歌搜索，post请求就是可以通过界面，请求浏览器做某些事，比如百度翻译界面。

## 爬取百度翻译

首先对界面和数据进行分析，假如我们点击了谷歌搜搜，然后在里面搜索dog英文单词，弄明白下面几个问题

- 依次点击这些文件，并进行观察
- 观察是什么请求，post还是get请求
- 数据如何提交，以及如何通过接口获取数据

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyps4d8bmj324g12unaf.jpg)

首先依次点击并查看返回的XHR文件信息

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyqc3o6adj31900k2tc7.jpg)

看到应该就是这个文件，可以看出是post请求，并且返回头里面显示的返回数据类型是json数据。

然后代码如下：

```
import requests

url = r'https://fanyi.baidu.com/#en/zh/'
param = {
'query': 'dog'
}
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.post(url=url, headers=header, data=param)
res.json()
```

**运行后可以发现，其实查询的网页并不是这个。怎么解决呢？**

因为网页是动态的，如果我们动态搜索，可以发现XHR中多了很多sug文件

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyqk9v6i5j31xw0z4gy8.jpg)

查看文件可以发现，这个文件里的信息才是我想要的

<img src="http://ww1.sinaimg.cn/large/005KJzqrgy1goyqlwyas7j30li0jo76v.jpg" alt="image.png" style="zoom: 50%;" />

修改代码后如下：

```
import requests
url = r'https://fanyi.baidu.com/sug'
param = {
'kw': 'dog'
}
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.post(url=url, headers=header, data=param)

text = res.json()

# json数据如下
{'errno': 0,
 'data': [{'k': 'dog', 'v': 'n. 狗; 蹩脚货; 丑女人; 卑鄙小人 v. 困扰; 跟踪'},
  {'k': 'DOG', 'v': 'abbr. Data Output Gate 数据输出门'},
  {'k': 'doge', 'v': 'n. 共和国总督'},
  {'k': 'dogm', 'v': 'abbr. dogmatic 教条的; 独断的; dogmatism 教条主义; dogmatist'},
  {'k': 'Dogo', 'v': '[地名] [马里、尼日尔、乍得] 多戈; [地名] [韩国] 道高'}]}
```

- 注意post参数data，相当于get的参数params
- 如果网页返回的数据是json，则可以不用text属性得到数据，直接使用.json方法，然后导入json模块处理json数据

**网页翻译一般都是post请求，可以动态和服务器进行交互，一般都是ajax请求。所以我们可以打开控制台的XHR进行抓包。**

**json数据可以根据下面这样查看：**

- 点击XHR抓到的文件，然后返回的数据是json数据
- 点击文件对应的response，就可以看到返回的json数据

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyqw9m1dnj31060gsjt9.jpg)

## 爬取豆瓣电影

首先分析网页：https://movie.douban.com/typerank?type_name=%E5%96%9C%E5%89%A7&type=24&interval_id=100:90&action=

搜索豆瓣电影，然后点击进去，找到排行榜。

- 打开控制台，打开XHR准备进行抓包

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyr69keubj323e0zcwqj.jpg)

- 不断向下翻页，可以看到多刷新了几个文件

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyr7q50gtj321i0bg0vx.jpg)

- 依次点击文件，并且查看文件信息，可以发现请求的url，还有返回的数据类型是json，以及请求的参数

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyrc61ef2j31980oqjwq.jpg)

**先不管下面请求的参数是什么，写到代码中观察下**

```
import requests

url = r'https://movie.douban.com/j/chart/top_list'
param = {
'type': '24',
'interval_id': '100:90',
'start': '20',
'limit': '20',
}
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.get(url=url, headers=header, params=param)

res.json()

# 可以看到返回的json数据
[{'rating': ['9.2', '50'],
  'rank': 21,
  'cover_url': 'https://img3.doubanio.com/view/photo/s_ratio_poster/public/p2409467410.jpg',
  'is_playable': False,
  'id': '26946612',
  'types': ['剧情', '喜剧', '爱情'],
  'regions': ['中国大陆'],
  'title': '狐妖小红娘剧场版：王权富贵',
  'url': 'https://movie.douban.com/subject/26946612/',
  'release_date': '2016-05-20',
  'actor_count': 7,
```

- 接下来可以调节参数，看看参数的意义是什么
- limit是限制获取的电影数量，star是起始电影

## 爬取肯德基餐厅查询

地址：http://www.kfc.com.cn/kfccda/storelist/index.aspx

刚打开的界面如下，可以发现什么东西都没有：

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyrzi8k4nj31vy17o1fw.jpg)

注意观察此时的网页地址，如果我输入北京，然后点击搜索，网页没有发生变化，说明浏览器和服务器的数据交换方式是动态的ajax请求。

**可以发现网址确实没有发生变化。**

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goys2rkxmuj31w411q4o9.jpg)

- 首先打开控制台，然后点击XHR进行抓包，在搜索框中输入关键字后，点击查询，可以看到捕获到的文件

<img src="/Users/mikizhu/Library/Application Support/typora-user-images/image-20210327212426981.png" alt="image-20210327212426981" style="zoom:50%;" />

- 查看文件信息，可以发现请求的网站，请求方式是post，返回的数据类型是text，不是json格式，请求的关键字参数。

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goys7hugjej30vo0t6q76.jpg)

接下来就可以编写代码了

```
import requests

url = r'http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx'
param = {
'keyword': '北京',
'pageIndex': '1',
'pageSize': '10',
'cname': '',
'pid':'',
}
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.post(url=url, headers=header, data=param)

with open('text.text', 'w') as f:
    f.write(res.text)
```

**然后会出现一些问题，网页返回的是-1000：http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=keyword**

我们发现请求的网页是这个，如果按照以前的方式，url为问号之前的，那么会出现问题

**其实这里的url就是这个网页，只不过用了一个变量keyword代替而已。**

```
url = r'http://www.kfc.com.cn/kfccda/ashx/GetStoreList.ashx?op=keyword' # 将url变换成这个即可
param = {
'keyword': '北京',
'pageIndex': '1',
'pageSize': '10',
'cname': '',
'pid':'',
}
```

## 药监总局爬虫

地址：http://scxk.nmpa.gov.cn:81/xk/

任务：爬取所有页数的每个公司的生产许可信息，第1/372页，15条/页，总共【5570】条数据

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytjyud7tj31vc0h6n1d.jpg)

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytkm711qj31m40vyh4o.jpg)

- 首先分析网页界面

打开控制台，进行抓包，点击all，可以看到浏览器抓到的所有包，xk/和浏览器地址很像，我们打开这个文件看看具体内容。

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytn1opdkj32720quwpo.jpg)

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytpvfv2nj326q12ikbt.jpg)

可以看看是不是可以直接使用get该地址获得网页的所有内容，MAC系统按command+f进行搜索，windows按crtl+f进行搜索

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytvkitjmj31b20q47e9.jpg)

- response就是get请求可以获得到的信息，可以发现网页中的公司都不在该页面中，所以这些公司很有可能是ajax动态请求到的，点击XHR就可以找到数据

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goytylxo8pj318m0aq763.jpg)

- 点击文件查看信息，查看url，数据传输为json，参数中的page size很重要。

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1goyu0h2a9fj30z20t80x8.jpg)

**json数据如下，里面有个很重要的信息 ID，如果有公司的url，我们就可以直接爬取了，但是json数据中没有网页的url，只能通过其他方式获取URL**

```
"list":[{"ID":"ed59438f34ae47e794f4c7ee5137c1f7","EPS_NAME":"海南京润珍珠生物技术股份有限公司","PRODUCT_SN":"琼妆20160001","CITY_CODE":"311","XK_COMPLETE_DATE":{"date":25,"day":0,"hours":0,"minutes":0,"month":3,"nanos":0,"seconds":0,"time":1619280000000,"timezoneOffset":-480,"year":121},"XK_DATE":"2026-04-25","QF_MANAGER_NAME":"海南省药品监督管理局","BUSINESS_LICENSE_NUMBER":"91460000294121210Y","XC_DATE":"2021-04-25","NUM_":1},{"ID":"5eb10afc74a2462c8e86652ec8d90a48","EPS_NAME":"无锡邦士立39f48609e97c24c7d3a94d9","EPS_NAME":"汕头市博美化妆品有限公司","PRODUCT_SN":"粤妆20160264","CITY_CODE":null,"XK_COMPLETE_DATE"
```

**多点击几个公司后，我们可以发现：他们只有ID是不同的**

```
http://scxk.nmpa.gov.cn:81/xk/itownet/portal/dzpz.jsp?id=327b7ce0c2214b6ea502f5cb00c0d1a9
http://scxk.nmpa.gov.cn:81/xk/itownet/portal/dzpz.jsp?id=5eb10afc74a2462c8e86652ec8d90a48
```

**因此可以通过ID来构造URL，通过for循环获得所有页数的数据**

