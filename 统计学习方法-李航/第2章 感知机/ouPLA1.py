import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 创建一个数据集，X有两个特征，y={-1，1}
X, y = make_blobs(n_samples=500, centers=2, random_state=6)
y[y==0] = -1
plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
plt.xlabel("feature_1")
plt.ylabel("feature_2")
plt.show()


class PerceptronDuality():
    def __init__(self):
        self.alpha = None
        self.bias = None
        self.W = None

    def fit(self, x_train, y_train, learning_rate=1, n_iters=100, plot_train=True):
        print("开始训练...")
        num_samples, num_features = x_train.shape
        self.alpha = np.zeros((num_samples,))
        self.bias = 0

        # 计算 Gram 矩阵
        gram = np.dot(x_train, x_train.T)

        while True:
            error_count = 0
            for idx in range(num_samples):
                inner_product = gram[idx]
                y_idx = y_train[idx]
                distance = y_idx * (np.sum(self.alpha * y_train * inner_product) + self.bias)
                # 如果有分类错误点，修正 alpha 和 bias，跳出本层循环，重新遍历数据计算，开始新的循环
                if distance <= 0:
                    error_count += 1
                    self.alpha[idx] = self.alpha[idx] + learning_rate
                    self.bias = self.bias + learning_rate * y_idx
                    break
                    # 数据没有错分类点，跳出 while 循环
            if error_count == 0:
                break
        self.W = np.sum(self.alpha * y_train * x_train.T, axis=1)
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
            plt.title("Dataset and Decision in Training(Duality)")
            plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, s=30, cmap=plt.cm.Paired)
            plt.plot(x_hyperplane, y_hpyerplane, color='g', label='Decision_Duality')
            plt.legend(loc='upper left')
            plt.show()

    def predict(self, x):
        if self.alpha is None or self.bias is None:
            raise NameError("模型未训练")
        y_predicted = np.sign(np.dot(x, self.W) + self.bias)
        return y_predicted

X_train = X[0:450]
y_train = y[0:450]
X_test = X[450:500]
y_test = y[450:500]

model_duality = PerceptronDuality()
model_duality.fit(X_train, y_train)

y_predict_duality = model_duality.predict(X_test)
accuracy_duality = np.sum(y_predict_duality == y_test) / y_test.shape[0]

print("对偶形式模型在测试集上的准确率: {0}".format(accuracy_duality))

print("对偶形式模型参数:")
print("W: {0}, bias: {1}".format(model_duality.W, model_duality.bias))