class MyIterator:
    def __init__(self):
        self.data = [1,2,3,4]
        self.curindex = -1
        self.less = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Add your logic here
        if self.less > 0:
            self.less = self.less - 1
        else:
            self.curindex = self.curindex + 1
            if self.curindex >= len(self.data):
                raise StopIteration
            self.less = self.data[self.curindex] - 1
        data = self.data[self.curindex]
        return data


print("")
my_iterator = MyIterator()
try:
    while True:
        item = next(my_iterator)
        print("==", item)
        # Do something with item
except StopIteration:
    # 迭代结束，执行相应的处理
    print("捕获到 StopIteration，遍历结束")
    pass

