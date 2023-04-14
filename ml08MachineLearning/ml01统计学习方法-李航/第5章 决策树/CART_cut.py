# -*-coding:utf-8-*-
# LANG=en_US.UTF-8
# CART 算法
# 文件名：CART.py
#

import sys
import math
import copy

dict_all = {
    # 1: 青年；2：中年；3：老年
    '_age': [
        1, 1, 1, 1, 1,
        2, 2, 2, 2, 2,
        3, 3, 3, 3, 3,
    ],

    # 0：无工作；1：有工作
    '_work': [
        0, 0, 1, 1, 0,
        0, 0, 1, 0, 0,
        0, 0, 1, 1, 0,
    ],

    # 0：无房子；1：有房子
    '_house': [
        0, 0, 0, 1, 0,
        0, 0, 1, 1, 1,
        1, 1, 0, 0, 0,
    ],

    # 1：信贷情况一般；2：好；3：非常好
    '_credit': [
        1, 2, 2, 1, 1,
        1, 2, 2, 3, 3,
        3, 2, 2, 3, 1,
    ],
}

# 0：未申请到贷款；1：申请到贷款
_type = [
    0, 0, 1, 1, 0,
    0, 0, 1, 1, 1,
    1, 1, 1, 1, 0,
]


# 二叉树结点
class BinaryTreeNode(object):
    def __init__(self, name=None, data=None, left=None, right=None, father=None):
        self.name = name
        self.data = data
        self.left = left
        self.right = left
        self.father = father


# 二叉树遍历
class BTree(object):
    def __init__(self, root=0):
        self.root = root

    # 中序遍历
    def inOrder(self, treenode):
        if treenode is None:
            return
        self.inOrder(treenode.left)
        print(treenode.name, treenode.data)
        self.inOrder(treenode.right)


# 获得种类中中每个特征的个数，以及该特征中_type = 1的个数 和 其他特征中_type = 1的个数
# 输入：字典中的当前种类的字典，列表 _type，待分析种类列表中的元素序号
# 输出字典：{ '特征': [特征的个数, 该特征中_type = 1(能贷到款)的个数, 其他种特征type = 1的个数] }
# eg，对于 _age：
#   因为其青中老年个 5 个，且青年中能带到款的有2个，中年和老年能贷到款的分别为3个和4个，所以输出：
#       {'1': [5, 2, 7], '2': [5, 3, 6], '3': [5, 4, 5]}
def get_value_type_num(_data, _type_list, num_list):
    value_dict = {}
    tmp_type = ''
    tmp_item = ''

    for num in num_list:
        item = str(_data[num])
        if tmp_item != item:
            if item in value_dict.keys():
                value_dict[item][0] = value_dict[item][0] + 1
                if _type_list[num] == 1:
                    value_dict[item][1] = value_dict[item][1] + 1
            else:
                if _type_list[num] == 1:
                    value_dict[item] = [1.0, 1.0, 0.0]
                else:
                    value_dict[item] = [1.0, 0.0, 0.0]
                tmp_item = item
        else:
            value_dict[item][0] = value_dict[item][0] + 1
            if _type_list[num] == 1:
                value_dict[item][1] = value_dict[item][1] + 1

    for num1 in range(len(value_dict)):
        for num2 in range(len(value_dict)):
            if num1 == num2: continue
            value_dict[value_dict.keys()[num1]][2] += value_dict[value_dict.keys()[num2]][1]

    return value_dict


# 获得种类中不同特征包含的元素序号
# 如：对应 dict_all 中的 _age，其包含青中老年，若 num_list 为 [0..15]，则输出：
#   {'1': [0, 1, 2, 3, 4], '2': [5, 6, 7, 8, 9], '3': [10, 11, 12, 13, 14]}
def get_value_type_no(data, data_type, num_list):
    value_dict = {}
    tmp_item = ''

    for num in num_list:
        item = str(data[data_type][num])
        if tmp_item != item:
            if item in value_dict.keys():
                value_dict[item].append(num)
            else:
                value_dict[item] = [num, ]
        else:
            value_dict[item].append(num)

    return value_dict


# 使用 gini 获得最优切分点
def get_cut_point_by_gini(_dict_all, _type_list, num_list, threshold):
    target_type = ''
    target_feature = ''
    target_gini = 1000000.0

    for data_key in _dict_all:
        value_dict = get_value_type_num(_dict_all[data_key], _type_list, num_list)
        tmp_feature = ''
        gini = 1000000.0
        # 通过计算当前种类的每一个特征的 gini 值，来获得该种类中 gini 最小的那个特征
        for value_key in value_dict.keys():
            all_feature_num = len(_dict_all[data_key])
            this_feature_num = value_dict[value_key][0]
            other_feature_num = all_feature_num - this_feature_num
            this_feature_yes_num = value_dict[value_key][1]
            other_feature_yes_num = value_dict[value_key][2]
            # 计算 gini
            tmp_gini = float('%.2f' % \
                             ( \
                                         (
                                                 (this_feature_num / all_feature_num) * \
                                                 2 * \
                                                 (this_feature_yes_num / this_feature_num) * \
                                                 (1 - this_feature_yes_num / this_feature_num) \
                                             ) + \
                                         (
                                                 (other_feature_num / all_feature_num) * \
                                                 2 * \
                                                 (other_feature_yes_num / other_feature_num) * \
                                                 (1 - other_feature_yes_num / other_feature_num) \
                                             ) \
                                 ))
            # 获得该种类中 gini 最小的那个特征
            if float(gini) - tmp_gini > 0.0:
                gini = tmp_gini
                tmp_feature = value_key

            if gini < threshold:
                return data_key, tmp_feature, 'over'

        # 通过对比所有种类中 gini 最小的特征，来获得 gini 最小的特征的种类, 该种类以及该种类的特征就是切分点
        if float(target_gini) - float(gini) > 0.0:
            target_type = data_key
            target_feature = tmp_feature

    return target_type, target_feature, 'continue'


# CART 算法
def CART(data, type_list, threshold):
    # 进行分类
    def classify(root, note_name, note_data, note_type):
        # 将'特征可能值名字'追加到 root.name 中
        # 将[样本序号的列表]合并到 root.data 中
        root.name.append(note_name)
        root.data.extend(note_data)

        # note_type=='exit' 意味着当前的数据全部属于某一类，不用在分类了
        if not data or note_type == 'exit':
            return

        target_type, target_feature, step = get_cut_point_by_gini(data, type_list, note_data, threshold)

        feature_dict = get_value_type_no(data, target_type, note_data)

        # 从样本集合中将该特征删除
        del data[target_type]

        # 准备左子节点和右子节点，节点的 name 和 data 是个空列表
        root.left = BinaryTreeNode([], [])
        root.right = BinaryTreeNode([], [])

        # 计算“特征字典”中各个集合中是属于“能贷贷款”的多还是“不能贷贷款”的多
        # 如果是前者：
        #   递归调用 classify，形成左子节点
        # 如果是后者：
        #   递归调用 classify，形成右子节点
        for key in feature_dict.keys():
            num_yes = 0;
            num_no = 0
            for num in feature_dict[key]:
                if type_list[num] == 1:
                    num_yes = num_yes + 1
                elif type_list[num] == 0:
                    num_no = num_no + 1
                else:
                    print('ERROR: wrong type in _type')
                    exit()

            note_type = 'not_exit'
            if num_yes == 0 or num_no == 0 or step == 'over':
                note_type = 'exit'

            if key == target_feature:
                classify(root.left, '%s:%s' % (target_type, key), feature_dict[key], note_type)
            else:
                classify(root.right, '%s:%s' % (target_type, key), feature_dict[key], note_type)

        return root

    tmp_list = []
    for num in range(len(dict_all.keys())):
        tmp_list.append(num)
    return classify(BinaryTreeNode([], []), 'root', tmp_list, 'not_exit')


class cost_complexity_pruning_parm(object):
    def __init__(self, sum_num):
        # 一共有多少个元素
        self.sum_num = sum_num
        # 某个节点的元素数
        self.node_num = 0.0
        # 某节点的叶子节点数量
        self.leaf_node_num = 0.0
        # 某节点的"错误分类"的元素数量
        self.node_data_error_num = 0.0
        # R(Tt)
        self.Rtt = 0.0
        # 节点的误差率增益值 g(t) 的字典，格式是{'节点名字': 节点的误差率增益值}
        self.error_rate_gain_dict = {}

    # 计算 R(Tt)
    # 参数：self, 该节点的"错误分类"的元素数量, 该节点的元素数
    def count_Rtt(self, node_item_num, node_err_num):
        self.Rtt = self.Rtt + ((node_err_num / node_item_num) * (node_item_num / self.sum_num))

    # 制作误差率增益值 g(t) 的字典
    # g(t) = R(t) - R(Tt) / ( |NTt| - 1 )
    # 参数：self, key, 该节点的"错误分类"的元素数量, 该节点的元素数
    def make_error_rate_gain_value_dict(self, key, node_item_num, node_err_num):
        rt = node_err_num / self.sum_num
        pt = node_item_num / self.sum_num
        Rt = rt * pt
        NTt = self.leaf_node_num
        self.error_rate_gain_dict[key] = float('%.3f' % float((Rt - self.Rtt) / (NTt - 1)))

    def print_error_rate_gain(self):
        print(self.error_rate_gain_dict)



def get_error_rate_gain_dict(dict_all_pruning, type_list, tree, cls):
    # 对某个节点求其误差率增益值
    def analyze_node(node, node_name, cls):
        # 如果是叶子节点，则叶子节点数 + 1，并计算 R(Tt)
        if not node.left and not node.right:
            cls.leaf_node_num = cls.leaf_node_num + 1
            dict_key = node.name[0].split(':')[0]
            value_dict = get_value_type_num(dict_all_pruning[dict_key], type_list, node.data)
            dict_key = node.name[0].split(':')[1]
            cls.count_Rtt(value_dict[dict_key][0], value_dict[dict_key][0] - value_dict[dict_key][1])
            return

        # 后续遍历
        analyze_node(node.left, None, cls)
        analyze_node(node.right, None, cls)
        # 如果遍历到 back_order 传进来的 node，则计算其“误差率增益值”
        if node.name[0] == node_name:
            dict_key = node.name[0].split(':')[0]
            # 获得 get_value_type_num 返回的字典(里面包含了该节点的元素总数和"正确分类"的元素数)
            value_dict = get_value_type_num(dict_all_pruning[dict_key], type_list, node.data)

            # 计算"错误分类"的元素数
            dict_key = node.name[0].split(':')[1]
            cls.make_error_rate_gain_value_dict(node.name[0], value_dict[dict_key][0],
                                                value_dict[dict_key][0] - value_dict[dict_key][1])
            return cls.leaf_node_num

    # 后续遍历决策树
    def back_order(node, cls):
        # 如果是叶子节点，则返回
        if not node.left and not node.right: return

        back_order(node.left, cls)
        back_order(node.right, cls)
        # 如果是根节点，则返回
        if node.name[0] == 'root': return

        cls.leaf_node_num = 0
        # 反之，求该结点的误差率增益值
        analyze_node(node, node.name[0], cls)

    back_order(tree.root, cls)


def cost_complexity_pruning(dict_all_pruning, type_list, tree, cls):
    # 进行剪枝
    def pruning(node, target_node_name):
        if not node.left and not node.right: return
        if node.name[0] == target_node_name:
            node.left = None
            node.right = None
            return

        pruning(node.left, target_node_name)
        pruning(node.right, target_node_name)

    # 获得误差率增益值 g(t) 的字典
    get_error_rate_gain_dict(dict_all_pruning, type_list, tree, cls)
    # cls.print_error_rate_gain()

    # 找出误差率增益值最小的节点
    min_error_rate_gain = 10000.0
    min_error_rate_gain_node = ''
    for key in cls.error_rate_gain_dict.keys():
        error_rate_gain = cls.error_rate_gain_dict[key]
        if error_rate_gain < min_error_rate_gain:
            min_error_rate_gain = error_rate_gain
            min_error_rate_gain_node = key

    pruning(tree.root, min_error_rate_gain_node)


# 阈值
# 如果使用 threshold = 0.3，那在使用 house 将样本数据分类后就停止了
# threshold = 0.3
threshold = 0.1
dict_all_cart = copy.deepcopy(dict_all)
root = CART(dict_all_cart, _type, threshold)
bt = BTree(root)
bt.inOrder(bt.root)
print('\n--------------\n')

# 这一步应该使用训练数据
dict_all_pruning = copy.deepcopy(dict_all)
cost_complexity_pruning(dict_all_pruning, _type, bt,
                        cost_complexity_pruning_parm(len(dict_all_pruning[dict_all_pruning.keys()[0]])))
bt.inOrder(bt.root)

# 剪枝前
#       root
#       /  \
# house:1  house:0
#           /  \
#      work:1  work:0
#
# 剪枝后(因为只有一个非叶子节点"house:0"，所以只能剪这个节点了)
#       root
#       /  \
# house:1  house:0
# 当然，这里剪这个不适合，因为剪枝前的决策树既不复杂也完全划分了样本数据，不过这里仅仅是实现剪枝算法，所以不考虑决策树适不适合剪枝。
# 顺便一提，"剪枝前的决策树在未用完种类的情况下完全划分了样本数据"可以作为适不适合剪枝的判断条件之一。
