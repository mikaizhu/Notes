## selenium操作说明

chrome driver 下载地址:https://chromedriver.chromium.org/downloads 

参考笔记：http://note.youdao.com/s/RVk8ocik

**到时候补充下：**

- 坐标操作
- js脚本执行

### 知识点补充

- 执行js程序

```
excute_script('jsCode')
```

- 前进后退

```
back()
forward()
```

- 根据属性名定位元素

```
# 先根据tag名找到所有元素
for i in find_elements_by_tag_name():
  if i.get_attribute('att_name') == '':
    do....
```

