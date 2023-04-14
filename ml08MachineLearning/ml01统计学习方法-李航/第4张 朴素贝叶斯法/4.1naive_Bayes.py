def loadDataSet():  # 加载表4.1数据
    x1 = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    x2 = ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']
    Y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    return x1, x2, Y


def nb(x1, x2, Y):  # 朴素贝叶斯算法部分
    x1_s = set(x1)  # 特征1去重：建立特征一的词典
    x2_s = set(x2)  # 特征二去重
    p(x1, x1_s, Y)  # 计算特征一分别属于1跟-1的概率，就是就是书上1，3两行
    p(x2, x2_s, Y)  # 计算特征二属于1，-1概率，书上2，4两行


def p(xj, x_s, Y):
    for x in x_s:  # 对每一个特征
        xcount = 0
        _xcount = 0
        for i in range(15):  # 对每一个样本
            if Y[i] == 1 and xj[i] == x:  # 再y=1时，每个特征出现次数
                xcount += 1
            elif Y[i] == -1 and xj[i] == x:  # y=-1时，每个特征出现次数
                _xcount += 1

        print('x=%s,Y=%d,p=%f' % (x, 1, xcount / Y.count(1)))  # 打印概率
        print('x=%s,Y=%d,p=%f' % (x, -1, _xcount / Y.count(-1)))


x1, x2, Y = loadDataSet()
nb(x1, x2, Y)
