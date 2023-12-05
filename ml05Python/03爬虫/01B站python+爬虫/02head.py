import requests
headers = {
    'User-Agent': 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'
}
response = requests.get("http://books.toscrape.com/", headers=headers)
print(response)  # <Response [200]>  Response类型的实例，代表服务器返回给我们的响应
print(response.status_code)      # 200
print(response.ok)      # True
if response.ok:
    print(response.text)
else:
    print("请求失败！")



