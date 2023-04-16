import requests
from bs4 import BeautifulSoup
import time
import random

urls = [
    f"https://www.cnblogs.com/sitehome/p/{page}"
    for page in range(1, 100 + 1)
]

def craw1(url):
    r = requests.get(url)
    print("<<< ", url, len(r.text), r.status_code, " >>>")

def craw(url):
    # print("craw url: ", url)
    r = requests.get(url)
    #time.sleep(random.random()*10)
    return r.text

def parse(html, url=None):
    # class="post-item-title"
    # print("     parse", url)
    soup = BeautifulSoup(html, "html.parser")
    links = soup.find_all("a", class_="post-item-title")
    result_list = [(link["href"].encode('utf-8').decode('utf-8'), link.get_text().encode('utf-8').decode('utf-8')) for link in links]
    return result_list

if __name__ == "__main__":
    for result in parse(urls[2], craw(urls[2])):
        print(result)

