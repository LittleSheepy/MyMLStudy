import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import copy
from pltGif import pltGif

# 创建一个数据集，X有两个特征，y={-1，1}
PLA_NUM = 50
PLA_NUM_TEST = 2
X, y = make_blobs(n_samples=PLA_NUM, centers=2, random_state=6)
y[y==1] = -1
y[y==0] = 1
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.show()
history = []  # 用来记录每次更新过后的w,b
#history.append([copy.copy(w), b])  # 将每次更新过后的w,b记录在history数组中
hisDot = []
hisL = []

class PerceptronRaw():
    def __init__(self):
        self.W = None;
        self.bias = None;
        self.history = []  # 用来记录每次更新过后的w,b
        self.hisDot = []

    def fit(self, x_train, y_train, learning_rate=0.05, n_iters=100, plot_train=True):
        print("开始训练...")
        num_samples, num_features = x_train.shape
        self.W = np.array([-0.1, 0.1])
        self.bias = 1
        self.history.append([copy.copy(self.W), self.bias])

        while True:
            erros_examples = []
            erros_examples_y = []
            distanceSum = 0

            w_ = np.sqrt(np.sum(np.array(self.W) ** 2))
            # 查找错误分类的样本点
            for idx in range(num_samples):
                example = x_train[idx]
                y_idx = y_train[idx]
                # 计算距离
                distance = y_idx * (np.dot(example, self.W) + self.bias)
                if distance <= 0:
                    erros_examples.append(example)
                    erros_examples_y.append(y_idx)
                    distanceSum += -distance/w_
            print(distanceSum)
            if len(erros_examples) == 0:
                break;
            else:
                # 随机选择一个错误分类点，修正参数
                random_idx = np.random.randint(0, len(erros_examples))
                choosed_example = erros_examples[random_idx]
                choosed_example_y = erros_examples_y[random_idx]
                self.W = self.W + learning_rate * choosed_example_y * choosed_example
                #self.bias = self.bias + learning_rate * choosed_example_y
                self.history.append([copy.copy(self.W), self.bias])
                self.hisDot.append(choosed_example)
        print("训练结束")

        # 绘制训练结果部分
        if plot_train is True:
            x_hyperplane = np.linspace(2, 10, 8)
            slope = -self.W[0] / self.W[1]
            intercept = -self.bias / self.W[1]
            y_hpyerplane = slope * x_hyperplane + intercept

            plt.xlabel("feature_1")
            plt.ylabel("feature_2")
            plt.xlim((2, 10))
            plt.ylim((-12, 0))
            plt.title("Dataset and Decision in Training(Raw)")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
            plt.plot(x_hyperplane, y_hpyerplane, color='g', label='Decision_Raw')
            plt.legend(loc='upper left')
            plt.show()

    def predict(self, x):
        if self.W is None or self.bias is None:
            raise NameError("模型未训练")
        y_predict = np.sign(np.dot(x, self.W) + self.bias)
        return y_predict

X_train = X[0:(PLA_NUM-PLA_NUM_TEST)]
y_train = y[0:(PLA_NUM-PLA_NUM_TEST)]
X_test = X[(PLA_NUM-PLA_NUM_TEST):PLA_NUM]
y_test = y[(PLA_NUM-PLA_NUM_TEST):PLA_NUM]

# 实例化模型，并训练
model_raw = PerceptronRaw()

model_raw.fit(X_train, y_train)

# 测试，因为测试集和训练集来自同一分布的线性可分数据集，所以这里测试准确率达到了 1.0
y_predict = model_raw.predict(X_test)

accuracy = np.sum(y_predict == y_test) / y_predict.shape[0]
print("原始形式模型在测试集上的准确率: {0}".format(accuracy))

pltGif_ = pltGif(np.column_stack((X, y)), model_raw.history, model_raw.hisDot)
pltGif_.save()