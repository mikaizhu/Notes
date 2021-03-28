## xpath解析方式

- 安装模块

```
pip install lxml
```

- 实例化对象

```
from lxml import etree
tree = etree.parse(filePath) # filePath为html文件
tree = etree.HTML('page_text') # 或者直接用html数据
```

- 使用xpath表达式，获取网页元素

```
tree.xpath('xpath表达式')
```

## 实战测试

- **首先获取一个html界面**

```
import requests
url = r'https://zh.wikipedia.org/wiki/%E5%94%90%E8%AF%97'
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.get(url=url, headers=header)
with open('tangshi.html', 'w') as f:
    f.write(res.text)
```

网页源码如下：

```
<!DOCTYPE html>
<html class="client-nojs" lang="zh" dir="ltr">
<head>
<meta charset="UTF-8"/>
<title>唐诗 - 维基百科，自由的百科全书</title>
<script>document.documentElement.className="client-js";RLCONF={"wgBreakFrames":!1,"wgSeparatorTransformTable":["",""],"wgDigitTransformTable":["",""],"wgDefaultDateFormat":"zh","wgMonthNames":["","1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"],"wgRequestId":"66e28f7b-91e6-46c1-84b2-999623bc442b","wgCSPNonce":!1,"wgCanonicalNamespace":"","wgCanonicalSpecialPageName":!1,"wgNamespaceNumber":0,"wgPageName":"唐诗","wgTitle":"唐诗","wgCurRevisionId":63773642,"wgRevisionId":63773642,"wgArticleId":33130,"wgIsArticle":!0,"wgIsRedirect":!1,"wgAction":"view","wgUserName":null,"wgUserGroups":["*"],"wgCategories":["自2018年5月缺少注脚的条目","唐朝诗歌"],"wgPageContentLanguage":"zh","wgPageContentModel":"wikitext","wgRelevantPageName":"唐诗","wgRelevantArticleId":33130,"wgUserVariant":"zh","wgIsProbablyEditable":!0,"wgRelevantPageIsProbablyEditable":!0,"wgRestrictionEdit":
```

- xpath是通过层级定位来查找元素的。可以根据空格来判断层级关系

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gozl4424wgj321e0n8wr6.jpg)

比如我们现在要爬取唐诗的标题

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1gozl7k4cpmj30v604agm6.jpg)

可以看到title在head标签下，然后又在html标签下

- **xpath层级定位**

所以xpath表达式可以这样写，就像查找文件一样

​		/表示一个层级

​		//表示两个层级

```
/html/head/title
```

上面还可以这样写

```
/html//title # 找到在html层级，任意一个层级中的title
//title # 找到任意两个层级中的title
```

**完整代码如下**：

```
import requests
import lxml
from lxml import etree

url = r'https://zh.wikipedia.org/wiki/%E5%94%90%E8%AF%97'
header = {
    'User-Agent' : 'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
res = requests.get(url=url, headers=header)
tree = etree.HTML(res.text)
tree.xpath('/html/head/title')
```

**返回的结果为一个列表，里面为一个element对象**

```
[<Element title at 0x7fdca99b7ec0>]
```

- **根据属性定位元素**

```
<meta property="og:title" content="唐诗 - 维基百科，自由的百科全书">
```

写法如下：

```
tree.xpath('//meta[@property="og:title"]')
```

> Meta 是标签tag，用@来定位标签的属性

- **索引定位**

```
//meta[@property="og:title"]/p[1] # 只找到第一个，从1开始不是从0开始
//meta[@property="og:title"]/p # 表示找到这个属性的所有p标签
```

> 总之斜杠后面可以不断层级索引，[1]表示第一个元素标签

- **找到元素后，就可以获得对象里面的内容**

```
# 获取文本内容
//meta[@property="og:title"]/p[1]/text() # 注意一个斜杠和两个斜杠的区别
//meta[@property="og:title"]/p[1]//text() # 两个斜杠表示跳过一个，中间为任意的
```

> 使用的是xpath获取文本内容

```
# 获取属性内容<link rel="stylesheet" href="/w/load.php?lang=zh-cn&amp;modules=site.styles&amp;only=styles&amp;skin=vector">
# 比如要获取href里面的内容

/html/head/link[3]/@href
```

> 使用@href即可，即@加属性名

**可以直接使用谷歌浏览器复制xpath路径**

