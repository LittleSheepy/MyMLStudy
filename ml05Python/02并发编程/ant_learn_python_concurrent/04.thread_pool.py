import concurrent.futures
import blog_spider
from concurrent.futures import ThreadPoolExecutor, as_completed
from blog_spider import urls, craw, parse
import time

def test1():
    # craw
    with ThreadPoolExecutor() as pool:
        htmls = pool.map(craw, urls)
        htmls = list(zip(urls, htmls))
        for url, html in htmls:
            print(url, len(html))

    # parse
    with ThreadPoolExecutor() as pool:
        futures = {}
        for url, html in htmls:
            future = pool.submit(parse, html, url)
            futures[future] = url

        # print("parse submit over")

        # 顺序固定
        for future, url in futures.items():
            print(">>>", url, future.result())

        # 顺序不定
        # for future in as_completed(futures):
        #     url = futures[future]
        #     print(url, future.result())


def test2():
    # craw
    with ThreadPoolExecutor() as pool:
        htmls = {}
        for url in urls:
            html = pool.submit(craw, url)
            htmls[html] = url

    # parse
    with ThreadPoolExecutor() as pool:
        futures = {}
        # 顺序固定
        for html, url in htmls.items():
            future = pool.submit(parse, html.result(), url)
            futures[future] = url

        # print("parse submit over")

        # 顺序固定
        for future, url in futures.items():
            result = future.result()
            #print("顺序固定", url)


def test3():
    # craw
    with ThreadPoolExecutor() as pool:
        htmls = {}
        for url in urls:
            html = pool.submit(craw, url)
            htmls[html] = url

    # parse
    with ThreadPoolExecutor() as pool:
        futures = {}
        # 顺序不定
        for html in as_completed(htmls):
            url = htmls[html]
            future = pool.submit(parse, html.result(), url)
            futures[future] = url

        # print("parse submit over")

        # 顺序不定
        for future in as_completed(futures):
            result = future.result()
            url = futures[future]
            #print("顺序不定", url)

if __name__ == '__main__':
    start = time.time()
    test2()
    end = time.time()
    print("test2 顺序固定 cost:", end - start, "seconds")
    start = time.time()
    test3()
    end = time.time()
    print("test3 顺序不定 cost:", end - start, "seconds")

