import numpy as np


class MultinomialNB(object):
    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = None
        self.classes = None
        self.conditional_prob = None
        self.predict_prob = None
        '''
        fit_class:是否学习类的先验概率，False则使用统一的先验
        class_prior:类的先验概率，如果指定，则先验不能根据数据调整

    '''

    def fit(self, x, y):
        # 计算类别y的先验概率
        self.classes = np.unique(y)

        if self.class_prior == None:  # 先验概率没有指定
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0 / class_num for i in range(class_num)]
            else:
                self.class_prior = {}
                for d in self.classes:
                    c_num = np.sum(np.equal(y, d))
                    self.class_prior[d] = (c_num + self.alpha) / (float(len(y) + class_num * self.alpha))
        # print(self.class_prior)
        # 计算条件概率------多项式
        self.conditional_prob = {}  # {x1|y1:p1,x2|y1:p2,.....,x1|y2:p3,x2|y2:p4,.....}
        y = list(y)
        for yy in self.class_prior.keys():
            y_index = [i for i, label in enumerate(y) if label == yy]
            # print(y_index)#标签的先验概率
            for i in range(len(x)):
                x_class = np.unique(x[i])
                for c in list(x_class):
                    x_index = [x_i for x_i, value1 in enumerate(list(x[i])) if value1 == c]
                    xy_count = len(set(x_index) & set(y_index))
                    pkey = str(c) + '|' + str(yy)
                    self.conditional_prob[pkey] = (xy_count + self.alpha) / (
                                float(len(y_index)) + len(list(np.unique(x[i]))))
        return self

    def predict(self, X_test):  # 此处只能对一个样本输入测试，加循环可以多个样本一个测试，高斯模型有实现
        self.predict_prob = {}
        for i in self.classes:
            self.predict_prob[i] = self.class_prior[i]

            for d in X_test:
                tkey = str(d) + '|' + str(i)
                self.predict_prob[i] = self.predict_prob[i] * self.conditional_prob[tkey]
        label = max(self.predict_prob, key=self.predict_prob.get)
        return label


class GaussionNB(object):  # 计算条件概率的方法不一样
    def __init__(self, fit_prior=True):
        self.fit_prior = fit_prior
        self.class_prior = None
        self.classes = None
        self.mean = None
        self.var = None
        self.predict_prob = None

    def fit(self, x, y):
        # 计算类别y的先验概率
        self.classes = np.unique(y)

        if self.class_prior == None:  # 先验概率没有指定
            class_num = len(self.classes)
            if not self.fit_prior:
                self.class_prior = [1.0 / class_num for i in range(class_num)]
            else:
                self.class_prior = {}
                for d in self.classes:
                    c_num = np.sum(np.equal(y, d))
                    self.class_prior[d] = (c_num) / (float(len(y)))
        # print(self.class_prior)
        # 计算条件概率------高斯
        self.mean = {}
        self.var = {}
        y = list(y)
        for yy in self.class_prior.keys():
            y_index = [i for i, label in enumerate(y) if label == yy]
            for i in range(len(x)):
                x_class = [x[i][j] for j in y_index]
                pkey = '特征' + str(i) + '|' + '类别' + str(yy)
                mean = np.mean(x_class)
                var = np.var(x_class)
                self.mean[pkey] = mean
                self.var[pkey] = var
        return self


    def _calculat_prob_gaussion(self, mu, sigma, x):
        prob = (1.0 / (sigma * np.sqrt(2 * np.pi)) *
                np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))
        return prob


    def predict(self, X_test):
        labels = []
        self.predict_prob = []
        predict_prob_ = {}
        for d in range(len(X_test)):
            x_test = X_test[d]
            # print(x_test)
            for yy in self.class_prior.keys():
                predict_prob_[yy] = self.class_prior[yy]
                for i, x in enumerate(list(x_test)):
                    tkey = '特征' + str(i) + '|' + '类别' + str(yy)
                    mu = self.mean[tkey]
                    sigma = self.var[tkey]
                    prob = self._calculat_prob_gaussion(mu, sigma, x)
                    predict_prob_[yy] = predict_prob_[yy] * prob

            print(predict_prob_)
            new_predict_prob_ = predict_prob_.copy()
            self.predict_prob.append(new_predict_prob_)
            label = max(predict_prob_, key=predict_prob_.get)
            labels.append(label)
        return labels

X = [[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3],[1,2,2,1,1,1,2,2,3,3,3,2,2,3,3]]
#X = X.T
y = [-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1]

mnb = GaussionNB()
mnb.fit(X,y)

labels = mnb.predict([[3,2]])
print(labels)