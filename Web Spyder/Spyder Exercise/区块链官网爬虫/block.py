import os
import requests
import pandas as pd

headers = {
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}

pages = 20 # 这里设置要爬取的页数
method = []
block = []
age = []
from_ = []
to_ = []
value = []
txn_free = []

for page in range(1, pages+1):
    print(f'正在爬取第{page}页...')
    url = f'https://etherscan.io/txs?a=0x6BEcAb24Ed88Ec13D0A18f20e7dC5E4d5b146542&p={page}'
    res = requests.get(url, headers=headers)
    tree = etree.HTML(res.text) # 或者直接用html数据
    txn_hash = tree.xpath('//*[@class="myFnExpandBox_searchVal"]/text()') # 获取文本内容
    for i in range(1, len(txn_hash) + 1):
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[3]/span/@title'
        method.extend(tree.xpath(xpath))
    
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[4]/a/text()'
        block.extend(tree.xpath(xpath))
    
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[6]/span/text()'
        age.extend(tree.xpath(xpath))
    
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[7]/span/text()'
        if tree.xpath(xpath):
            from_.extend(tree.xpath(xpath))
        else:
            from_.append('')
    
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[9]/span/a/text()'
        if tree.xpath(xpath):
            to_.extend(tree.xpath(xpath))
        else:
            # 这里会出现问题，所以用else
            xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[9]/span/span/a/text()'
            if tree.xpath(xpath):
                to_.extend(tree.xpath(xpath))
            else:
                to_.append('')
    
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[10]//text()'
        temp = ''.join(tree.xpath(xpath)) # 这里使用字符串进行拼接
        value.append(temp)
    
   
        xpath = f'//*[@id="paywall_mask"]/table/tbody/tr[{i}]/td[11]/span/text()'
        temp = '.'.join(tree.xpath(xpath))
        txn_free.append(temp)

# 检查数据长度是否一致
print(len(method), len(block), len(age), len(from_), len(to_), len(value), len(txn_free))

data = {
    'method':method,
    'block':block,
    'age':age,
    'from_':from_,
    'to_':to_,
    'value':value,
    'txn_free':txn_free,
    }

df = pd.DataFrame(data)

df.to_csv('bloc.csv') # 数据保存
