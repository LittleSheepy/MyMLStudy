from bs4 import BeautifulSoup
import requests


headers = {
    'User-Agent': 'Opera/8.0 (Windows NT 5.1; U; en)'
}
response = requests.get("http://books.toscrape.com/", headers=headers)
content = response.text
soup = BeautifulSoup(content, "html.parser")
# print(soup)
# print(soup.p)
# 查找价格
all_prices = soup.findAll("p",attrs={"class": "price_color"})
print(len(all_prices))
for price in all_prices:
    # print(price)
    # print(price.string[2:])
    pass
# 查找书名
all_titles = soup.findAll("h3")
print(len(all_titles))
print(all_titles[0])
for title in all_titles:
    # all_links = title.findAll("a")
    # print(len(all_links))
    # for link in all_links:
    #     print(link.string)
    #因为只有一个a
    link = title.find("a")
    print(link.string)




