import queue
import blog_spider
from io import TextIOWrapper
import time
import random
import threading

# 生产者
def do_craw(url_queue:queue.Queue, html_queue:queue.Queue):
    while True:
        url = url_queue.get()
        html = blog_spider.craw(url)
        html_queue.put(html)
        print(threading.current_thread().name, f"craw {url}",
              "url_queue.size=", url_queue.qsize())
        time.sleep(random.randint(1,2))


# 消费者
def do_parse(html_queue:queue.Queue, f_out:TextIOWrapper):
    while True:
        html = html_queue.get()
        results = blog_spider.parse(html)
        for result in results:
            try:
                f_out.write(str(result) + "\n")
            except:
                pass
        print(threading.current_thread().name, f"results.size", len(results),
              "html_queue.size=", html_queue.qsize())
        time.sleep(random.randint(1,2))

if __name__ == '__main__':
    url_queue = queue.Queue()
    html_queue = queue.Queue()
    for url in blog_spider.urls:
        url_queue.put(url)

    for idx in range(3):
        t = threading.Thread(target=do_craw, args=(url_queue, html_queue),
                             name=f"craw{idx}")
        t.start()

    f_out = open("02data.text", "w")
    for idx in range(2):
        t = threading.Thread(target=do_parse, args=(html_queue, f_out),
                             name=f"parse{idx}")
        t.start()
    while html_queue.qsize():
        pass
    # f_out.close()