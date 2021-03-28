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

**xpath 表达式说明：**



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

