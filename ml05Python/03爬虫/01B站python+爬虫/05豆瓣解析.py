from bs4 import BeautifulSoup
import requests


headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.5735.289 Safari/537.36'
}

for start_num in range(0, 250, 25):
    print(start_num)
    response = requests.get(f"http://movie.douban.com/top250?start={start_num}", headers=headers)
    # print(response)
    # if response.ok:
    #     print(response.text)

    content = response.text
    soup = BeautifulSoup(content, "html.parser")

    all_titles = soup.findAll("span", attrs={"class": "title"})
    print(len(all_titles))
    for title in all_titles:
        title_string = title.string
        if "/" not in title_string:
            print(title_string)





