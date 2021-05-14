#!/Users/mikizhu/miniconda3/envs/py38_env/bin/python3
import requests
import re

headers = {
'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 11_1_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.192 Safari/537.36'
}
params = {
        'username':'2070436044',
        'password':'12180030',
        'dllt':'userNamePasswordLogin',
        'execution':'e1s1',
        '_eventld':'submit',
        'rmShown':1,
        }
login_url = 'https://authserver.szu.edu.cn/authserver/login?service=http%3A%2F%2Fehall.szu.edu.cn%2Flogin%3Fservice%3Dhttp%3A%2F%2Fehall.szu.edu.cn%2Fnew%2Findex.html'
res = requests.get(url=login_url, headers=headers, params=params, allow_redirects=False)
#print(res.text)

p = re.compile('href="(.*?)"')
print(p.findall(res.text))

url = p.findall(res.text)
res = requests.get(url=url, headers=headers, params=params, allow_redirects=True)
print(res.text)
