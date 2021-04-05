尝试登陆网站：https://so.gushiwen.cn/user/login.aspx?from=http://so.gushiwen.cn/user/collect.aspx

账号：747876457@qq.com

密码：zwl123456

因为有些网站要登陆后才能获取到里面的信息，除了使用selenium进行登陆，还可以使用requests进行模拟浏览器登陆

- 首先打开登陆界面，进行抓包，找到请求的界面，注意打开preserve功能

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gozn1v4h5tj327y0mkn5t.jpg)

> 谷歌开发者工具里面这个**preserve log** ：保留请求日志，跳转页面的时候勾选上，可以看到跳转前的请求，也可适用于chrome开发者工具抓包的问题.

- 可以找到login文件，发现里面提交的网址和参数

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gozn4h716qj31ag0viwnl.jpg)

请求的URL为：

```
https://so.gushiwen.cn/user/login.aspx?from=http%3a%2f%2fso.gushiwen.cn%2fuser%2fcollect.aspx
```

- 开始模拟登陆

```
import requests

url = r'https://so.gushiwen.cn/user/login.aspx?from=http%3a%2f%2fso.gushiwen.cn%2fuser%2fcollect.aspx'
param = {
'__VIEWSTATE':'ujE9lDTHp7rhe/ajEut/gasyEMg4OoYjedrkLbOiFW4PIIJlLyfuBWhdMRMzecxrOAEYE1exKK1o5Yr2BBgsodu82mXBSAohVL51nfGtFf96xgmbqSf8jQNGuBA=',
'__VIEWSTATEGENERATOR': 'C93BE1AE',
'from': 'http://so.gushiwen.cn/user/collect.aspx',
'email': '747876457@qq.com',
'pwd': 'zwl123456',
'code': 'zkr8',
'denglu': '登录',
}
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.post(url=url, headers=header, data=param)
print(res.status_code)

with open('text.txt', 'w') as f:
    f.write(res.text)
```

**按理来说，使用这个code就OK了，但是不行，是因为我每次提交请求，验证码都会变化，所以最好使用第三方识别，然后进行模拟登陆，这种模拟登陆方式不好**

**总结**：**还是selenium比较香～**

另一种方式是使用selenium登陆后，得到cookie，然后通过cookie来访问里面的任何内容，cookie就相当于身份证，你输入账号密码后就会获得这个身份证，访问里面的内容都要通过身份证

可以有多种方式处理cookie

- cookie怎么来的

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gp01dp89a4j326o0xox4o.jpg)

打开浏览器进行抓包，点击login文件可以看到响应有很多set cookie，**所以cookie其实是由服务器给我们创建的**

- 手动处理cookie

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gp01gkbkjhj326o13ktkj.jpg)

当我们登陆之后，要访问登陆之后界面网址，就必须在请求头上即headers上添加上cookie，我们可以手动添加，但这样的坏处是，有些cookie是有时间限制的，过了一段时间后就会没用了。手动处理cookie代码如下：

```
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36',
    'Cookie':'xxxxxx'
}
res = requests.post(url=url, headers=header, data=param)
print(res.status_code)
```

- 自动处理cookie， session对象的使用

使用session对象来处理cookie比较方便。cookie产生的方式主要有1. 登陆的时候服务器会给我们设置cookie。2. 登陆完后，请求网页的时候，也会有cookie发生变化。

```
session = requests.Session() # 实例化对象

# 然后使用session模块，发送所有请求
res = session.post(url=url, headers=header, data=param)
res = session.get()
print(res.status_code)
```

自动处理cookie的方式，代码如下：

```
header = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36',
}
param = {
    'callback': 'fetchJSON_comment98',
    'productId': '5561746',
    'score': '3',
    'sortType': '5',
    'pageSize': '10',
    'isShadowSku': '0',
    'fold': '1',
}
cookie = {
    'Cookie': 'shshshfpa=cb4386b6-c0f2-68a1-b156-2236f499ee30-1590065631; shshshfpb=dXHF9pqH0l8XV8dgbxTlNEQ%3D%3D; __jdu=15900656294791955369661; user-key=3e85b9e4-c7cf-43fd-9e1a-756e9776cab0; cn=0; __jdc=122270672; areaId=19; ipLoc-djd=19-1607-3155-0; shshshfp=83f76a7577a1d3cdcb7f20cd9a99ba87; __jdv=122270672|github.com|-|referral|-|1617588163257; jwotest_product=99; __jda=122270672.15900656294791955369661.1590065629.1617593724.1617596142.26; 3AB9D23F7A4B3C9B=NYA7Y2IYQW7V35YN3PSDHABICJZ5GIKPEEIE6XO7TSEUVYNHVZ7CFQHTY2RYGTGNNEFG2YNVNV5ZYJC36L2IOMHRSM; shshshsID=273af6177249142f26677cc915a91991_2_1617597325824; __jdb=122270672.2.15900656294791955369661|26.1617596142; JSESSIONID=DED3BD27A82165E1DED7C5656BFD65D1.s1'
} # cookie单独设置为dic

session = requests.Session()
cookie = requests.utils.cookiejar_from_dict(cookie)
session.cookies = cookie
res = session.get(url=good_comments_url, headers=header, data=param)
```

- request和selenium获取cookie的案例

参考：https://blog.csdn.net/weixin_40444270/article/details/80593058

```
import requests
from selenium import webdriver
import time

c = driver.get_cookies()
cookies = {}
# 获取cookie中的name和value,转化成requests可以使用的形式
# 这里得到的cookie好像是json的形式
for cookie in c:
    cookies[cookie['name']] = cookie['value']
    
response = requests.get(url='https://cart.jd.com/cart.action', headers=headers, cookies=cookies)
```

