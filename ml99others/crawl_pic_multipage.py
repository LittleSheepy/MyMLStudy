import re
import requests
import random
import ssl
ssl.create_default_https_context = ssl._create_unverified_context  #验证SSL


def find_img(html):
    tag = '[a-zA-z]+://[^\\s]*[a-zA-z]'             #核心在这里，正则匹配出符合标准的URL
    # \\s其实是\s,转义
    imgre = re.compile(tag)
    img_url_list = imgre.findall(html)
    return set(img_url_list)

def getHTMLText(url,ua):         #获取HTML文本
    try:
        r = requests.get(url,headers=ua)
        #自定义请求头部
        r.encoding = r.apparent_encoding
        return r.text #获得正文内容
    except:
        return "产生异常"

num=1
def save_img(img_list):       #保存文件
    root = "/Users/wyx/Desktop/pic1/"  #保存路径
    global num
    for img_url in img_list:
        try:
            if len(img_url) > 100:   #弥补之前正则式的缺陷
                pic = requests.get(img_url,timeout=10)  #设置timeout，避免等待
                print('正在下载：'+ str(num) +'.jpg...')

            else:
                continue
        except requests.exceptions.ConnectionError:
            print('URL错误')
            continue
        path = root  + str(num) + '.jpg'  #文件命名为: 序号.jpg
        num = num + 1
        # print(path)
        fp = open(path,'wb')
        fp.write(pic.content)
        fp.close()

if __name__ == "__main__":
    aotourl = 'http://www.bizhi88.com/s/116077/'
    url = aotourl    #这个URL是可以分析出来的
    # User-Agent
    ua = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}

    i = 1
    for i in range(3):  #先下到第5页,不想下太多
        url = aotourl + str(i) + '.html' #每一页的内容
        temp = getHTMLText(url, ua) #获取http请求的正文内容
        if temp != "产生异常":
            imgs = find_img(temp)#从获取图片
            save_img(imgs)#保存图片