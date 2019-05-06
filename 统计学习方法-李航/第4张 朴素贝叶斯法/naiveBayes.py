
import numpy as np

"""
@:param features : 输入的特征，array类型
@:param label : 输入features对应的label，array类型
@:param input: 给定的一组特征，判断属于哪一个类别
"""
def naive_bayes_classifiers(features, label, input_data):
    #计算各个label的概率
    label_category = list(set(label))
    label_number = len(label_category)
    label_prob = {}
    for i in range(0, label_number):
        label_prob[label_category[i]] = len(label[label==label_category[i]])/len(label)
    print(label_prob) #{1: 0.6, -1: 0.4}

    #2、计算label条件下每一类特征中不同特征取值的条件概率

    len_features = len(features)
    features_set = {}
    #2.1、计算每一类特征里有哪些取值
    for i in range(0,len_features):
        features_set[i] = list(set(features[i]))
    print(features_set) #{0: ['2', '3', '1'], 1: ['L', 'S', 'M']},表明有两类特征
    #2.2、计算不同label(-1/1)情况下每类特征不同取值的概率P，key为feature+label,
    #比如，01-1表示在Y=-1的情况下第一类特征且取值为1的条件概率P(X0=1|Y=-1)
    #011表示在Y=1的情况下第一类特征且取值为1的条件概率P(X0=1|Y=1)
    features_label_prob = {}
    prob_nums = len(features_set[0])*len(features)*label_number
    # print(prob_nums) # 计算一共多少种条件概率的组合
    for i in range(0, len(features)):
        for j in range(0, len(features_set[i])):
            for k in range(0, label_number):
                label_select = label[label==label_category[k]]
                denominator = len(label_select)
                # print(denominator)
                label_index = np.where(label==label_category[k])
                object_feature = np.array(features[i])[label_index]
                # print(object_feature)
                numberator_value = object_feature[object_feature==features_set[i][j]]
                numberator = len(numberator_value)
                features_label_prob_key = str(i)+str(features_set[i][j])+str(label_category[k])
                features_label_prob[features_label_prob_key] = numberator/denominator
                # print(features_label_prob_key) # 比对key，分子numberator, 分母denominator 没有问题
                # print(numberator)
                # print(denominator)
    print(features_label_prob)

    #3、计算给定input_data的类别概率
    calc_label_prob = {}
    multi = 1
    for i in range(0, label_number):
        calc_label_prob[label_category[i]] = label_prob[label_category[i]]
        # print(calc_label_prob[label_category[i]])
        for j in range(0, len(input_data)):
            key = str(j)+str(input_data[j])+str(label_category[i])
            # print(key)
            # print(features_label_prob[key])
            multi = multi * features_label_prob[key]
        calc_label_prob[label_category[i]] = calc_label_prob[label_category[i]]*multi
        multi = 1

    # print(calc_label_prob) #{1: 0.02222222222222222, -1: 0.06666666666666667}

    #返回概率最大的label
    output_label = max(calc_label_prob,key=calc_label_prob.get)
    # print(output_label)
    return output_label
if __name__ == '__main__':
    features = [[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
                ['S', 'M', 'M', 'S', 'S', 'S', 'M', 'M', 'L', 'L', 'L', 'M', 'M', 'L', 'L']]
    features = np.array(features)
    label = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    label = np.array(label)
    input_data = np.array([2, 'S'])
    output_label = naive_bayes_classifiers(features, label, input_data)
    print("the output label of input_data is:{}".format(output_label))
