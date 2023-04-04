#读取输入数据，并且转换为列表
# data1 = input()
data1 = "3 abc bca cab abc 1"
data1 = data1.split()
#获取单词的个数
n1 = data1[0]
#按字典排序的第几个兄弟词
n2 = data1[-1]
#获取输入的n个单词
data2 = data1[1:-2]
#获取兄弟词
data3 = data1[-2]

#用于存储兄弟词的数量
n3 = 0
#用于存储兄弟词
data4 = []

for word in data2:
    if word == data3:
        continue
    elif sorted(word) == sorted(data3):
        n3 = n3 + 1
        data4.append(word)
print(n3)
#将兄弟词按照字典排序
data5 = sorted(data4)
if int(n2)-1 < len(data5):
    print(data5[int(n2)-1])
