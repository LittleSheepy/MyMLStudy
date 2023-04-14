import matplotlib.pyplot as plt
import numpy as np
import operator

# dataPredict -- 用于分类的输入向量/测试数据
# k -- 选择最近邻的数目
def kNNclassify(dataPredict, dataSet, labels, k):
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(dataPredict, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #3. 排序并返回出现最多的那个类型
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

if __name__ == '__main__':
    dataSet = np.array([[1.0, 1.1], [0, 0], [1.0, 1.0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']

    print(kNNclassify([0.1, 0.1], dataSet, labels, 3))

    # 显示数据
    plt.figure()