import requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.289 Safari/537.36'
}
response = requests.get("http://movie.douban.com/top250/", headers=headers)
print(response)
if response.ok:
    print(response.text)


