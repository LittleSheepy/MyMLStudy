import re
import requests
import random
import ssl
ssl.create_default_https_context = ssl._create_unverified_context  #验证SSL

def find_img(html):
    tag = '"objURL":"(.*?)"'             #核心在这里，正则匹配出符合标准的URL
    imgre = re.compile(tag)
    img_url_list = imgre.findall(html)
    return img_url_list

def getHTMLText(url,ua):         #获取HTML文本
    try:
        r = requests.get(url,headers=ua)
        r.encoding = r.apparent_encoding
        return r.text
    except:
        return "产生异常"

def save_img(img_list,word):       #保存文件
    root = "/Users/wyx/Desktop/pic/"  #保存路径
    i = 91
    for img_url in img_list:
        try:
            pic = requests.get(img_url,timeout=10)  #设置timeout，避免等待
            print('正在下载'+word+str(i)+'.jpg...')
        except requests.exceptions.ConnectionError:
            print('URL错误')
            continue
        path = root + word + str(i)+'.jpg'    #文件命名为: 关键字 + 序列号（从1开始）.jpg
        # print(path)
        i = i+1
        fp = open(path,'wb')
        fp.write(pic.content)
        fp.close()

if __name__ == "__main__":
    baidu_img = 'http://image.baidu.com/search/index?tn=baiduimage&ps=1&ct=201326592&lm=-1&cl=2&nc=1&ie=utf-8&word='   #百度图片
    keyword = input('请输入要搜索的关键词： ')
    url = baidu_img + keyword    #这个URL是可以分析出来的
    #url="https://image.baidu.com/search/index?tn=baiduimage&ipn=r&ct=201326592&cl=2&lm=-1&st=-1&fm=result&fr=&sf=1&fmq=1572514569011_R&pv=&ic=&nc=1&z=&hd=&latest=&copyright=&se=1&showtab=0&fb=0&width=&height=&face=0&istype=2&ie=utf-8&sid=&word=%E5%8F%A3%E7%BD%A9%E7%9A%84%E4%BA%BA"
    # User-Agent
    ua = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100 Safari/537.36'}
    temp = getHTMLText(url, ua)
    if temp != "产生异常":
        # print(find_img(temp))
        imgs = find_img(temp)
        save_img(imgs,keyword)