import requests
import json

response = requests.get('http://www.baidu.com')
print("response.status_code:", response.status_code)  # 打印状态码
print(response.url)          # 打印请求url
print(response.headers)      # 打印头信息
print(response.cookies)      # 打印cookie信息
print("response.text:", response.text)  #以文本形式打印网页源码
print(response.content) #以字节流形式打印
print("response:", response)

data = {'name':'tom','age':'22'}

response = requests.post('http://httpbin.org/post', data=data)
#print(response.text)
r_json = response.json()
print(response.json())

r = requests.get('https://github.com/timeline.json')
r_j = r.json()
print(r.json()) # 需要先import json

import requests
import chardet
url = 'http://www.baidu.com/'
resp = requests.get(url)
enc = chardet.detect(resp.content)

#resp.encoding = enc

print(chardet.detect(resp.content))
print(resp.text)