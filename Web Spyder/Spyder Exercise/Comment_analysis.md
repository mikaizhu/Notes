**目前想法：爬取商品的好评和差评，并且打上标签，然后做情感分析**

目前找到的商品：

- 男衬衫 https://item.jd.com/100012784316.html#comment 差评100多
- 二手手机 https://item.jd.com/31472535611.html#none 差评80多
- 洗面奶 https://item.jd.com/5561746.html#none 差评 1w多

**初步代码：**

```
from selenium import webdriver
import time
import pickle

class Spyder:
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.path1 = r'https://item.jd.com/100012784316.html#comment' # 男衬衫，差评100+
        self.path2 = r'https://item.jd.com/31472535611.html#none' # 二手手机，差评80+
        self.path3 = r'https://item.jd.com/5561746.html#none' # 洗面奶，差评1W+
        self.comments = []
        self.labels = []

    def get_good_comment(self):
        res = self.driver.find_elements_by_class_name('comment-con')
        for _ in res:
            self.comments.append(_.text)
        self.labels.extend([1]*len(res)) # 因为是好评，所以标记为1
    
    def get_bad_comment(self):
        res = self.driver.find_elements_by_class_name('comment-con')
        for _ in res:
            self.comments.append(_.text)
        self.labels.extend([0]*len(res)) # 因为是好评，所以标记为1
    
    def get_path1_comments(self, star_page=1, end_page=7):
        self.driver.get(self.path1)
        time.sleep(2)
        self.driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[5]').click() # 点击商品评论标签
        time.sleep(1)
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a').click() # 点击好评标签
        time.sleep(2)
        for page in range(star_page, end_page):
            try: # 爬取好评
                self.driver.find_element_by_xpath(f'//*[@id="comment-4"]/div[12]/div/div/a[{page}]').click()
                time.sleep(1)
                self.get_good_comment()
                print(f'第{page}页好评爬取成功！')
            except:
                print(f'好评爬取失败...')
                
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a').click() # 点击差评标签
        for page in range(star_page, end_page):
            try: # 爬取差评
                time.sleep(1) # 因为怕元素没加载出来，所以这里睡眠2秒
                self.driver.find_element_by_xpath(f'//*[@id="comment-6"]/div[12]/div/div/a[{page}]').click()
                self.get_bad_comment()
                print(f'第{page}页差评爬取成功！')
            except:
                print(f'差评爬取失败...')
        self.save_result(self.comments, self.labels, comment_name='shirt_comments')
        print('数据保存成功！')
    
    def get_path2_comments(self, star_page=1, end_page=7):
        self.driver.get(self.path2)
        time.sleep(2)
        self.driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[4]').click() # 点击商品评论标签
        time.sleep(1)
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a').click() # 点击好评标签
        time.sleep(2)
        for page in range(star_page, end_page):
            try: # 爬取好评
                self.driver.find_element_by_xpath(f'//*[@id="comment-4"]/div[12]/div/div/a[{page}]').click()
                time.sleep(1)
                self.get_good_comment()
                print(f'第{page}页好评爬取成功！')
            except:
                print(f'好评爬取失败...')
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a').click() # 点击差评标签
        for page in range(star_page, end_page):
            try: # 爬取差评
                time.sleep(1) # 因为怕元素没加载出来，所以这里睡眠2秒
                self.driver.find_element_by_xpath(f'//*[@id="comment-6"]/div[12]/div/div/a[{page}]').click()
                self.get_bad_comment()
                print(f'第{page}页差评爬取成功！')
            except:
                print(f'差评爬取失败...')
        self.save_result(self.comments, self.labels, comment_name='phone_comments')
        print('数据保存成功！')

    def get_path3_comments(self, star_page=1, end_page=101):
        self.driver.get(self.path3)
        time.sleep(2)
        self.driver.find_element_by_xpath('//*[@id="detail"]/div[1]/ul/li[5]').click() # 点击商品评论标签
        time.sleep(1)
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[5]/a').click() # 点击好评标签
        time.sleep(2)
        for page in range(star_page, end_page):
            try: # 爬取好评
                self.driver.find_element_by_xpath(f'//*[@id="comment-0"]/div[12]/div/div/a[{page}]').click()
                time.sleep(1)
                self.get_good_comment()
                print(f'第{page}页好评爬取成功！')
            except:
                print(f'好评爬取失败...')
        self.driver.find_element_by_xpath('//*[@id="comment"]/div[2]/div[2]/div[1]/ul/li[7]/a').click() # 点击差评标签
        for page in range(star_page, end_page):
            try: # 爬取差评
                time.sleep(1) # 因为怕元素没加载出来，所以这里睡眠1秒
                self.driver.find_element_by_xpath(f'//*[@id="comment-6"]/div[12]/div/div/a[{page}]').click()
                self.get_bad_comment()
                print(f'第{page}页差评爬取成功！')
            except:
                print(f'差评爬取失败...')
        self.save_result(self.comments, self.labels, comment_name='phone_comments')
        print('数据保存成功！')
    
    def save_result(self, feature, label, comment_name, save_path='/Users/mikizhu/Desktop/'):
        assert isinstance(comment_name, str) # 判断输入的comment name是不是字符串类型
        assert isinstance(save_path, str)
        features_name = comment_name + '_feature.pkl'
        labels_name = comment_name + '_label.pkl'
        pickle.dump(feature, open(save_path + features_name, 'wb'))
        pickle.dump(label, open(save_path + labels_name, 'wb'))
    
    def quit(self):
        self.driver.quit()
        
# 实例化对象      
spyder = Spyder()

# 开始爬取评论
spyder.get_path1_comments()
spyder.get_path2_comments()
spyder.get_path3_comments()

# 关闭浏览器
spyder.quit()
```

