#!/Users/mikizhu/miniconda3/envs/py38_env/bin/python3
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time
from selenium.webdriver.support.ui import Select

def login(username, password, login_url):
    driver.get(login_url)
    driver.find_element_by_id('username').send_keys(username)
    driver.find_element_by_id('password').send_keys(password)
    time.sleep(2)
    driver.find_element_by_css_selector('#casLoginForm > p:nth-child(5) > button').click()
    time.sleep(3)
    # 从最近使用中找到体育馆预约按钮
    driver.find_element_by_xpath('//*[@id="widget-hot-01"]/div[1]/widget-app-item[1]/div/div/div[2]/div[1]').click()
    driver.implicitly_wait(5)

    # 点击进入服务按钮
    driver.find_element_by_xpath('//*[@id="ampDetailEnter"]').click()
    driver.implicitly_wait(5)
    # 因为浏览器会打开新标签，跳转到预约标签
    for i in driver.window_handles:
        driver.switch_to.window(i)
        if '体育场馆预约' == driver.title:
            break
    # 因为个人包场要鼠标悬停，所以使用selenium的悬停操作
    time.sleep(3)
    try:
        for i in driver.find_elements_by_tag_name('div'):
            if i.get_attribute('title') == '个人包场预约':
                i.click()
                break
    except:
        mouse = driver.find_element_by_xpath('/html/body/header/header[1]/div/div/div[4]/div[3]')
        ActionChains(driver).move_to_element(mouse).perform() 
    
def go_book_index():
    # 点击个人包场预约
    time.sleep(3)
    driver.find_element_by_css_selector('body > div.bh-header-navMoreBox > a:nth-child(1) > div').click()
    print('登陆成功!正在查询场地')


def searching_playground(book_day='20'):
    print('Runing Searching Playground Function...')
    try:
        # 选择南区
        driver.find_element_by_xpath('//*[@id="sportCQSelect"]/span[1]/div').click();time.sleep(1)
        # 选择羽毛球选项
        driver.find_element_by_xpath('//*[@id="sportItemsSelect"]/span[3]/div').click();time.sleep(1)
        # 选择南区羽毛球场选项
        driver.find_element_by_xpath('//*[@id="sportCGSelect"]/span/div').click();time.sleep(1)
        # 进入日期选项，点击日期选择框
        for i in driver.find_elements_by_tag_name('input'):
            if i.get_attribute('autocomplete') == 'off':
                i.click()
                break
        time.sleep(3)
        # 设置查找日期的xpath, 这个xpath每天都要手动更换。点击时间20号,设置好时间后要等待一段时间
        for i in driver.find_elements_by_tag_name('td'):
            if i.text == book_day:
                i.click()
        # 点击完日期后要等待搜索，设置等待时间为4s
        time.sleep(4)
        # 如果查询不到场地，返回状态0，可以查询到，则返回状态1
        if driver.find_element_by_xpath('//*[@id="selSjd"]/div/div[2]/div[1]/div/div').text == '无可预约的时间段':
            print('没有可预约的时间段，继续等待！')
            return 0
        else:
            print('Searching Playground Success!')
            return 1
    except Exception as e:
        print('Searching Playground Function Error..')
        print(e)
        return 2
def keep_searching(statue=0, interval=1, book_day='20'):
    '''
    statue:状态码,默认为0，表示还需要查找
    interval:时间间隔，默认刷新时间为1分钟
    '''
    # 搜索场地的循环
    # 如果没找到场地，会一直执行这个函数
    while True:
        if statue == 1:
            break
        else:
            # 如果状态为0，表示查找失败，需要重新找
            # 如果状态为2，说明有元素没找到，则只要刷新网页即可。
            driver.refresh()
            if statue == 0:
                sec = interval*60
                while sec != 0:
                    print(f'搜索等待时间，还剩下{sec}s ！', end='\r')
                    sec -= 1
                    time.sleep(1)
            if statue == 2:
                time.sleep(2)
            statue = searching_playground(book_day=book_day)
    return statue
def start_booking(statue, book_ground='D3', book_day='20', book_time='09:00-10:00'):
    print('Runging Start Booking Function...')
    try:
        print(statue)
        if not statue:
            return
        driver.implicitly_wait(10)
        # 选择时间
        find = None
        for i in driver.find_elements_by_tag_name('select'):
            if i.get_attribute('name') == 'sjdselect':
                find = i.text.split()
                print(find)
                break
        # 这里的时间是你想设置的运动时间
        if book_time not in find:
            print('没搜索到想要预定的时间！')
            return 0
        print('找到预约时间，开始预约！')
        # 选择时间
        driver.implicitly_wait(5)
        sl = Select(driver.find_element_by_name('sjdselect'))
        sl.select_by_value(book_time)
        # 选择好时间后要等待一段时间
        for i in driver.find_elements_by_tag_name('button'):
            if i.text == '确定':
                i.click()
        driver.implicitly_wait(15)
        # 选择运动场地
        statue = 0
        playground_list = driver.find_element_by_id('sportCDSelect').text.split()
        for ground in playground_list:
            # 如果有想预约的场地，则选择，否则默认选择出现的第一个
            if book_ground in ground:
                idx = playground_list.index(ground) + 1
                driver.find_element_by_xpath(f'//*[@id="sportCDSelect"]/span[{idx}]').click();time.sleep(1)
                statue = 1
        if not statue:
            driver.find_element_by_xpath('//*[@id="sportCDSelect"]/span[1]').click();time.sleep(1)
        # 进入日期选项，点击日期选择框
        for i in driver.find_elements_by_tag_name('input'):
            if i.get_attribute('autocomplete') == 'off':
                i.click()
                break
        # 选择时间,20号
        for i in driver.find_elements_by_tag_name('td'):
            if i.text == book_day:
                i.click()
                break
        # 选择可预约时间段
        time_list = driver.find_element_by_xpath('//*[@id="sportItemForm"]/div/div[12]/div[1]').text.split()
        count = 0
        for i in time_list:
            if book_time in i:
                count = time_list.index(i)
        driver.find_element_by_xpath(f'//*[@id="sportItemForm"]/div/div[12]/div[1]/div/div/label[{count}]').click()
        time.sleep(1)
        # 选择最大人数
        for i in driver.find_elements_by_tag_name('input'):
            if i.get_attribute('data-caption') == '参与人数':
                i.send_keys('6')
                break
        driver.find_element_by_xpath('//*[@id="submitBooking"]').click()
        # 点击提交预约
        print('预约成功！')
        time.sleep(10)
        driver.quit()
        print('Play Ground Booking Success!')
        return 1
    except Exception as e:
        print('Start Booking Function Error:')
        print(e)
        driver.refresh()
        return 2

def alert_pay(statue, username, password, login_url):
    driver = webdriver.Chrome()
    if not statue:
        # 如果状态为0，则直接退出
        return
    login(username, password, login_url)
    # 进入我的预约
    time.sleep(2)
    for i in driver.find_elements_by_tag_name('div'):
        if i.get_attribute('title') == '我的预约':
            i.click()
            break
    sec = 3000
    while sec != 0:
        print(f'请快点付款，还剩下{sec}s 时间！', end='\r')
        sec -= 1
        time.sleep(1)
    driver.quit()

login_url = 'https://authserver.szu.edu.cn/authserver/login?service=http%3a%2f%2fehall.szu.edu.cn%2flogin%3fservice%3dhttp%3a%2f%2fehall.szu.edu.cn%2fnew%2findex.html'
driver = webdriver.Chrome()
username = '2070436044' # 设置账号密码
password = '12180030'
book_time = '09:00-10:00' # 预约的具体时间，小时
password = 'xxxxxxxx'
book_time = '21:00-22:00' # 预约的具体时间，小时
book_day = '20' # 预约的时间，日
book_ground = 'D2' # 设置预定的场地，如果没找到，则默认预定出现的第一个
interval = 0.5 # 设置每1分钟刷新一次
login(username, password, login_url=login_url)
go_book_index() # 去到预定界面

statue = 0 # 状态码如果为1，则说明运行成功，或者预约到了，为0则继续预约
while True:
    if statue == 1:
        break
    else:
        # 如果有场地，则statue为1，继续查找
        statue = keep_searching(statue=statue, interval=interval, book_day=book_day)
        # 如果有场地，但是没有想要的时间，则继续查找
        time.sleep(2)
        statue = start_booking(statue=statue, book_time=book_time, book_ground=book_ground)
        if statue != 1:
            continue
        time.sleep(2)
        alert_pay(statue, username, password, login_url)
        break
        alert_pay(statue)
