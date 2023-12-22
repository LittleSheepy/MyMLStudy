from urllib.request import urlopen
import re
import urllib

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
'Cookie':'bid=wjbgW95-3Po; douban-fav-remind=1; __gads=ID=f44317af32574b60:T=1563323120:S=ALNI_Mb4JL8QlSQPmt0MdlZqPmwzWxVvnw; __yadk_uid=hwbnNUvhSSk1g7uvfCrKmCPDbPTclx9b; ll="108288"; _vwo_uuid_v2=D5473510F988F78E248AD90E6B29E476A|f4279380144650467e3ec3c0f649921e; trc_cookie_storage=taboola%2520global%253Auser-id%3Dff1b4d9b-cc03-4cbd-bd8e-1f56bb076864-tuct427f071; viewed="26437066"; gr_user_id=7281cfee-c4d0-4c28-b233-5fc175fee92a; dbcl2="158217797:78albFFVRw4"; ck=4CNe; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1583798461%2C%22https%3A%2F%2Faccounts.douban.com%2Fpassport%2Flogin%3Fredir%3Dhttps%253A%252F%252Fmovie.douban.com%252Ftop250%22%5D; _pk_ses.100001.4cf6=*; __utma=30149280.1583974348.1563323123.1572242065.1583798461.8; __utmb=30149280.0.10.1583798461; __utmc=30149280; __utmz=30149280.1583798461.8.7.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/passport/login; __utma=223695111.424744929.1563344208.1572242065.1583798461.4; __utmb=223695111.0.10.1583798461; __utmc=223695111; __utmz=223695111.1583798461.4.4.utmcsr=accounts.douban.com|utmccn=(referral)|utmcmd=referral|utmcct=/passport/login; push_noty_num=0; push_doumail_num=0; _pk_id.100001.4cf6=06303e97d36c6c15.1563344208.4.1583798687.1572242284.'}
base_url = 'https://movie.douban.com/top250?start=%d&filter='
base_url = 'https://book.douban.com/top250?start=%d'


class spider_douban250(object):
        def __init__(self,url = None, start = 0, step = 25 , total = 250):
            self.durl = url
            self.dstart = start
            self.dstep =step
            self.dtotal = total

        def start_download(self):
            while self.dstart < self.dtotal:
                durl = self.durl%self.dstart
                print(durl)
                self.load_page(durl)
                self.dstart += self.dstep

        def load_page(self,url):
            req=urllib.request.Request(url=url,headers=headers)
            req = urlopen(req)
            if req.code != 200:
                return
            con = req.read().decode('utf-8')
            listli = re.findall(r'<table width="100%">(.*?)</table>', con,re.S)
            if listli:
                listli = listli[1:]
            else:
                return
            for li in listli:
                imginfo = re.findall(r'<img.*?>', li)
                if imginfo:
                    imginfo = imginfo[0]
                    #info = [item.split('=')[1].strip()[1:-1] for item in imginfo.split(' ')[2:4]]
                    info = imginfo.split(' ')[1].split('=')[1][1:-1]
                    self.load_img(info)

        def load_img(self, info):
            global num
            print("callhere load img:", info)
            req = urllib.request.Request(url=info, headers=headers)
            imgreq = urlopen(req)
            img_c = imgreq.read()
            path = r'D:\\test\\' + str(num) + '.jpg'
            print('path:', path)
            imgf = open(path, 'wb')
            imgf.write(img_c)
            imgf.close()
            num = num + 1
num = 0
spider = spider_douban250(base_url,start=0,step=25,total=250)
spider.start_download()