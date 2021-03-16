[TOC]

# 爬虫设计

**目前想法：爬取商品的好评和差评，并且打上标签，然后做情感分析**

目前找到的商品：

- 男衬衫 https://item.jd.com/100012784316.html#comment 差评100多
- 二手手机 https://item.jd.com/31472535611.html#none 差评80多
- 洗面奶 https://item.jd.com/5561746.html#none 差评 1w多

**完整代码如下：**

```
from selenium import webdriver
import time
import pickle
import logging

def set_logger(file_name):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y/%d/%m %I:%M:%S')

    # 设置记录文件名和记录的形式
    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(formatter)

    # 设置让控制台也能输出信息
    file_stream = logging.StreamHandler()
    file_stream.setFormatter(formatter)

    # 为记录器添加属性，第一行是让记录器只记录错误级别以上，第二行让记录器日志写入，第三行让控制台也能输出
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(file_stream)
    return logger

class Spyder:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.path1 = r'https://item.jd.com/100012784316.html#comment' # 男衬衫，差评100+
        self.path2 = r'https://item.jd.com/31472535611.html#none' # 二手手机，差评80+
        self.path3 = r'https://item.jd.com/5561746.html#none' # 洗面奶，差评1W+
        self.comments = []
        self.labels = []

    def get_comments(self, star_page, end_page, good_xpath=None, bad_xpath=None, good_comment=True):
        for page in range(star_page, end_page):
            try:
                res = self.driver.find_elements_by_class_name('comment-con')
                for _ in res:
                    self.comments.append(_.text)
                if good_comment:
                    self.labels.extend([1]*len(res)) # 因为是好评，所以标记为1
                    logger.info(f'第{page}页好评爬取成功！')
                else:
                    self.labels.extend([0]*len(res))
                    logger.info(f'第{page}页差评爬取成功！')
                
                time.sleep(2)
                pages = self.driver.find_element_by_link_text("下一页")
                self.driver.execute_script("arguments[0].click();", pages)
            except:
                self.driver.refresh()
                time.sleep(2)
                if good_comment:
                    self.click(good_xpath)
                    try:
                        self.get_comment(page, page+1, good_comment=True)
                    except:
                        logger.info(f'第{page}页好评爬取失败...')
                else:
                    self.click(bad_xpath)
                    try:
                        self.get_comment(page, page+1, good_comment=False)
                    except:
                        logger.info(f'第{page}页差评爬取失败...')
    
    def get_path1_comments(self, star_page=1, end_page=7):
        self.driver.get(self.path1)
        time.sleep(2)
        self.click('//*[@id="detail"]/div[1]/ul/li[5]') # 点击商品评论标签
        time.sleep(1)
        try:
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a') # 点击好评标签
        except:
            self.driver.refresh()
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a') # 继续点击好评标签
        time.sleep(2)
        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            good_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a',
            good_comment=True
        )

        time.sleep(2)       
        self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a') # 点击差评标签
        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            bad_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a',
            good_comment=False
        )
        self.save_result(self.comments, self.labels, comment_name='shirt_comments')
        logger.info('数据保存成功！')
    
    def get_path2_comments(self, star_page=1, end_page=7):
        self.driver.get(self.path2)
        time.sleep(2)
        self.click('//*[@id="detail"]/div[1]/ul/li[4]') # 点击商品评论标签
        time.sleep(1)
        try:
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a') # 点击好评标签
        except:
            self.driver.refresh()
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a') # 如果没加载出来，就刷新然后点击好评标签
        time.sleep(2)
        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            good_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a',
            good_comment=True
        )

        time.sleep(1)
        self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]') # 点击差评标签

        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            bad_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]',
            good_comment=False
        )

        self.save_result(self.comments, self.labels, comment_name='phone_comments')
        logger.info('数据保存成功！')

    def get_path3_comments(self, star_page=1, end_page=128):
        self.driver.get(self.path3)
        time.sleep(2)
        self.click('//*[@id="detail"]/div[1]/ul/li[5]') # 点击商品评论标签
        time.sleep(2)
        try:
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a') # 点击好评标签
            logger.info(f'正在爬取好评评论！')
        except:
            logger.info(f"重新尝试爬取好评！")
            self.driver.refresh() # 如果没有加载出评论，就重新刷新网页
            time.sleep(3) # 每次refresh要等待时间
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a')
        time.sleep(2)

        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            good_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a',
            good_comment=True
        )

        time.sleep(2)
        try:
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a') # 点击差评标签
        except:
            self.driver.refresh()
            time.sleep(3)
            self.click('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a') # 点击差评标签
            
        logger.info(f'正在爬取差评评论！')
        self.get_comments(
            star_page=star_page,
            end_page=end_page,
            bad_xpath='//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a',
            good_comment=False
        )

        self.save_result(self.comments, self.labels, comment_name='Facial_cleanser_comments')
        logger.info('数据保存成功！')
    
    def save_result(self, feature, label, comment_name, save_path='/Users/mikizhu/Desktop/'):
        assert isinstance(comment_name, str) # 判断输入的comment name是不是字符串类型
        assert isinstance(save_path, str)
        features_name = comment_name + '_feature.pkl'
        labels_name = comment_name + '_label.pkl'
        pickle.dump(feature, open(save_path + features_name, 'wb'))
        pickle.dump(label, open(save_path + labels_name, 'wb'))
    
    def quit(self):
        self.driver.quit()
    
    def click(self, xpath):
        # 设置两种方式进行点击，总有一个能成功
        try:
            self.driver.find_element_by_xpath(xpath).click()
        except:
            page = self.driver.find_element_by_xpath(xpath) # 点击差评标签
            page = self.driver.find_element_by_link_text(page.text)
            self.driver.execute_script("arguments[0].click();", page)
       
# 设置日志
logger = set_logger(file_name='comment.log') 
        
# 实例化对象      
spyder = Spyder()

# 开始爬取评论
time.sleep(2)
logger.info('='*10 + '正在爬取【衬衫】评论' + '='*10)
spyder.get_path1_comments()

time.sleep(2)
logger.info('='*10 + '正在爬取【手机】评论' + '='*10)
spyder.get_path2_comments()

time.sleep(2)
logger.info('='*10 + '正在爬取【洗面奶】评论' + '='*10)
spyder.get_path3_comments()

# 关闭浏览器
spyder.quit()
```



# 爬虫问题记录

**问题1:出现评论加载不出来的bug！**

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1golyhiv5haj31a80f4taq.jpg)

从图中可以看到，商品的评论有400万个，但是却显示暂无评价。

解决办法：刷新网页

输入下面代码即可，这里使用的是selenium

```
driver.refresh()
```

结果如图：

![image.png](http://ww1.sinaimg.cn/large/005KJzqrgy1golyjzu74tj31dm0jiwht.jpg)

**问题2:元素不可点击**

因为要爬取每一页的评论，可以用selenium模拟点击“下一页”这个元素，明明可以点击，但是代码却抱错提示不能点击

**报错如下：**

```
selenium.common.exceptions.ElementClickInterceptedException: Message: 
Element <a class=""> is not clickable at point 
(318.3000030517578,661.7999877929688)
```

**解决办法：**

参考：https://blog.csdn.net/WanYu_Lss/article/details/84137519

```
pages = self.driver.find_element_by_link_text("下一页")
self.driver.execute_script("arguments[0].click();", pages)
```

**问题3:在点击过程中，爬取频率过快，依旧会导致第一个问题**

如果评论没加载出来，就刷新，并从当前页从新爬取，但是刷新后，要重新点击好评或者差评标签

**解决办法：**

修改代码逻辑，原来的代码逻辑如下：

```
def get_comments(self, star_page, end_page, good_comment=True):
        for page in range(star_page, end_page):
            try:
            		res = self.driver.find_elements_by_class_name('comment-con')
                for _ in res:
                    self.comments.append(_.text)
                if good_comment:
                    self.labels.extend([1]*len(res)) # 因为是好评，所以标记为1
                    logger.info(f'第{page}页好评爬取成功！')
                time.sleep(2)
                pages = self.driver.find_element_by_link_text("下一页")
                self.driver.execute_script("arguments[0].click();", pages)
                
                else:
                    self.labels.extend([0]*len(res))
                    logger.info(f'第{page}页差评爬取成功！')
            except:
                if good_comment:
                    logger.info(f'第{page}页好评爬取失败...')
                else:
                    logger.info(f'第{page}页差评爬取失败...')
```

修改成如下后，爬取成功率大大提升：

```
def get_comments(self, star_page, end_page, good_comment=True):
        for page in range(star_page, end_page):
            try:
                res = self.driver.find_elements_by_class_name('comment-con')
                for _ in res:
                    self.comments.append(_.text)
                if good_comment:
                    self.labels.extend([1]*len(res)) # 因为是好评，所以标记为1
                    logger.info(f'第{page}页好评爬取成功！')
                else:
                    self.labels.extend([0]*len(res))
                    logger.info(f'第{page}页差评爬取成功！')
                
                time.sleep(2)
                pages = self.driver.find_element_by_link_text("下一页")
                self.driver.execute_script("arguments[0].click();", pages)
            except:
                self.driver.refresh()
                if good_comment:
                    try:
                        self.get_comment(page, page+1, good_comment=True)
                    except:
                        logger.info(f'第{page}页好评爬取失败...')
                else:
                    try:
                        self.get_comment(page, page+1, good_comment=False)
                    except:
                        logger.info(f'第{page}页差评爬取失败...')
```

**问题记录4:**

在点击元素的时候，如果使用xpath方法，容易出现bug，然后报错，说该元素不能点击。

**报错如下：**

```
selenium.common.exceptions.ElementClickInterceptedException: Message: element click intercepted: Element <a href="#none" clstag="shangpin|keycount|product|chaping_tuijianpaixu_eid=100^^tagid=ALL^^pid=20006^^sku=598699^^sversion=1001^^pageSize=1">...</a> is not clickable at point (815, 8). Other element would receive the click: <li data-tab="trigger" data-anchor="#shop-similar-promotion" clstag="shangpin|keycount|product|bendianhaopingshangpin">...</li>
  (Session info: chrome=89.0.4389.82)
```

最好的解决办法是：

- 先用xpath找到该元素的text
- 然后通过text定位点击该元素

**代码如下**

```
r = driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a')
pages = self.browser.find_element_by_link_text(r.text)
driver.execute_script("arguments[0].click();", pages)
```

# 数据分析

- 先对爬取结果读取，去重，查看爬了多少评论

```
res1 = pickle.load(open('/Users/mikizhu/Desktop/Facial_cleanser_comments_feature.pkl', 'rb'))
len(set(res1)) # 1994
res2 = pickle.load(open('/Users/mikizhu/Desktop/phone_comments_feature.pkl', 'rb'))
len(set(res2)) # 77
res3 = pickle.load(open('/Users/mikizhu/Desktop/shirt_comments_feature.pkl', 'rb'))
len(set(res3)) # 105
```

```
# 总共 2176 条评论
```

