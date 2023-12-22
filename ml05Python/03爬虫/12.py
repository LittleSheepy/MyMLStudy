import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

url = 'https://news.baidu.com/'
search_keyword = '阿里巴巴'

driver = webdriver.Chrome()
driver.get(url)

wait = WebDriverWait(driver, 20)  # 增加等待时间为20秒
search_box = wait.until(EC.presence_of_element_located((By.ID, 'ww')))
search_box.send_keys(search_keyword)

search_button = wait.until(EC.element_to_be_clickable((By.ID, 'su')))
search_button.click()

wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'result-op')))
html_code = driver.page_source
# re.findall(r'<div><h3 class="news-title_1YtI1 "><a href=".*?" target=".*?".*? aria-label="(.*?)"', html_code)
title_pattern = re.compile(r'<div><h3 class="news-title_1YtI1 "><a href=(.*?)</h3>.*?</div>')
publisher_pattern = re.compile(r'<span class="news-source_1BRPw">.*?>(.*?)</a>')

titles = re.findall(title_pattern, html_code)
publishers = re.findall(publisher_pattern, html_code)

for title, publisher in zip(titles, publishers):
    print(f'Title: {title} | Publisher: {publisher}')

driver.quit()