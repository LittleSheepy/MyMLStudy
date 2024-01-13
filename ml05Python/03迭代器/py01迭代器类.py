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


def next_test():
    my_iterator = MyIterator()
    try:
        while True:
            item = next(my_iterator)
            print("==", item)
    except StopIteration:
        print("捕获到 StopIteration，遍历结束")
        pass

def for_test():
    my_iterator = MyIterator()
    for item in my_iterator:
        print("==", item)




if __name__ == '__main__':
    print("")
    for_test()
