# --*-- coding:utf-8 --*--
import numpy as np


class Node:  # 结点
    def __init__(self, data, lchild=None, rchild=None):
        self.data = data
        self.lchild = lchild
        self.rchild = rchild


class KdTree:  # kd树
    def __init__(self):
        self.kdTree = None

    def create(self, dataSet, depth):  # 创建kd树，返回根结点
        if len(dataSet) == 0: return None
        dataSet = np.array(dataSet)
        m, n = np.shape(dataSet)  # 求出样本行，列
        midIndex = int(m / 2)  # 中间数的索引位置
        axis = depth % n  # 判断以哪个轴划分数据
        max_feat_ind_list = dataSet[:, axis].argsort()      # 进行排序
        sortedDataSet = dataSet[max_feat_ind_list]
        node = Node(sortedDataSet[midIndex])  # 将节点数据域设置为中位数，具体参考下书本
        leftDataSet = sortedDataSet[: midIndex]  # 将中位数的左边创建2改副本
        rightDataSet = sortedDataSet[midIndex + 1:]
        node.lchild = self.create(leftDataSet, depth + 1)  # 将中位数左边样本传入来递归创建树
        node.rchild = self.create(rightDataSet, depth + 1)
        return node

    def search(self, tree, x):  # 搜索
        self.nearestPoint = None  # 保存最近的点
        self.nearestValue = None  # 保存最近的值

        def travel(node, depth=0):  # 递归搜索
            if node != None:  # 递归终止条件
                n = len(x)  # 特征数
                axis = depth % n  # 计算轴
                if x[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
                    travel(node.lchild, depth + 1)
                else:
                    travel(node.rchild, depth + 1)

                # 以下是递归完毕后，往父结点方向回朔，对应算法3.3(3)
                distNodeAndX = np.sqrt(sum((x - node.data) ** 2))    # 目标和节点的距离判断
                if (self.nearestValue == None):  # 确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX
                elif (self.nearestValue > distNodeAndX):
                    self.nearestPoint = node.data
                    self.nearestValue = distNodeAndX

                print(node.data, depth, self.nearestValue, node.data[axis], x[axis])
                if (abs(x[axis] - node.data[axis]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                    if x[axis] < node.data[axis]:
                        travel(node.rchild, depth + 1)
                    else:
                        travel(node.lchild, depth + 1)

        travel(tree)
        return self.nearestPoint

if __name__ == '__main__':
    dataSet = [[2, 3],
               [5, 4],
               [9, 6],
               [4, 7],
               [7.1, 1],
               [7, 2]]
    x = [5, 3]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)
    print(kdtree.search(tree, x))
