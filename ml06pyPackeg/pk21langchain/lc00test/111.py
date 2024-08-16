def t():
    installs = [
        "1. pip install numpy",
        "3. 还可以试试豆瓣的源：pip install numpy -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com",
        "2. 如果装不了，就试试阿里的源：pip install numpy -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com",
    ]

    installs.sort(key=lambda i: i[0])
    for step in installs:
        items = step.split("：")
        print(items[0])
        if len(items) > 1:
            print(f"  {items[1]}")


if __name__ == '__main__':
    t()