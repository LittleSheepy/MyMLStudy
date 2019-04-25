# --*-- coding:utf-8 --*--
import numpy as np

# 结点
class Node:
    def __init__(self, data, dim=None, parent=None, lchild=None, rchild=None):
        self.data = data        # 结点的值(样本信息)
        self.dim = dim          # 结点的切分的维度(特征)
        self.parent = parent    # 父结点
        self.lchild = lchild    # 左子树
        self.rchild = rchild    # 右子树

# kd树
class KdTree:
    def __init__(self):
        self.kdTree = None

    def create(self, dataSet, depth, parentNode=None):  # 创建kd树，返回根结点
        if len(dataSet) == 0: return None
        dataSet = np.array(dataSet)
        m, n = np.shape(dataSet)                            # 求出样本行，列
        midIndex = m // 2                                   # 中间数的索引位置
        axis = depth % n                                    # 判断以哪个轴划分数据
        max_feat_ind_list = dataSet[:, axis].argsort()      # 进行排序
        sortedDataSet = dataSet[max_feat_ind_list]
        node = Node(sortedDataSet[midIndex], dim=axis, parent=parentNode)                # 将节点数据域设置为中位数，具体参考下书本
        leftDataSet = sortedDataSet[: midIndex]             # 将中位数的左边创建2改副本
        rightDataSet = sortedDataSet[midIndex + 1:]
        node.lchild = self.create(leftDataSet, depth + 1, node)   # 将中位数左边样本传入来递归创建树
        node.rchild = self.create(rightDataSet, depth + 1, node)
        self.kdTree = node
        return node

    # 找最近邻叶子节点
    def find_nearest_leaf(self, item, node=None):
        n = len(item)  # 特征数
        node = node if node != None else self.kdTree
        axis = node.dim
        if item[axis] < node.data[axis]:  # 如果数据小于结点，则往左结点找
            if node.lchild is None:return node
            return self.find_nearest_leaf(item, node.lchild)
        else:
            if node.rchild is None:
                if node.lchild is None: return node
                return node.lchild
            return self.find_nearest_leaf(item, node.rchild)

    def search(self, item):
        node = self.find_nearest_leaf(item)
        self.nearestPoint = node  # 保存最近的点
        self.nearestValue = np.sqrt(sum((item - node.data) ** 2))  # 保存最近的值
        def travel(cur_node, root):
            dis = np.sqrt(sum((item - cur_node.data) ** 2)) # 确定当前点，更新最近的点和最近的值，对应算法3.3(3)(a)
            if dis < self.nearestValue:
                self.nearestPoint = cur_node
                self.nearestValue = dis
            if cur_node == root:return
            parent = cur_node.parent
            if parent == None:return
            if (abs(item[parent.dim] - parent.data[parent.dim]) <= self.nearestValue):  # 确定是否需要去子节点的区域去找（圆的判断），对应算法3.3(3)(b)
                other_child = parent.lchild if parent.lchild != cur_node else parent.rchild  # 找另一个子树
                if other_child != None:
                    other_node = self.find_nearest_leaf(item, other_child)  # 找到最近的那个结点
                    if other_node != None:  # 空树
                        travel(other_node, other_child)
            travel(parent, root)
        travel(node, self.kdTree)
        return self.nearestPoint

if __name__ == '__main__':
    dataSet = [[2, 3], [5, 4], [9, 6], [4, 7], [8, 1], [7, 2]]
    x = [5, 3]
    kdtree = KdTree()
    tree = kdtree.create(dataSet, 0)
    print(kdtree.find_nearest_leaf(x).data)
    print(kdtree.search(x).data)