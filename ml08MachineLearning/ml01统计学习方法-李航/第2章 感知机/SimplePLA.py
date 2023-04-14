from random import randint
import numpy as np
import matplotlib.pyplot as plt


class TrainDataLoader:
    def __init__(self):
        pass
    def GenerateRandomData(self, count, gradient, offset):
        x1 = np.linspace(1, 5, count)
        x2 = gradient*x1 + np.random.randint(-10,10,*x1.shape)+offset
        dataset = []
        y = []
        for i in range(*x1.shape):
            dataset.append([x1[i], x2[i]])
            real_value = gradient*x1[i]+offset
            if real_value > x2[i]:
                y.append(-1)
            else:
                y.append(1)
        return x1,x2,np.mat(y),np.mat(dataset)


class SimplePerceptron:
    def __init__(self, train_data = [], real_result = [], eta = 1):
        self.w   =   np.zeros([1, len(train_data.T)], int)
        self.b   =   0
        self.eta =   eta
        self.train_data   = train_data
        self.real_result  = real_result
    def nomalize(self, x):
        if x > 0 :
            return 1
        else :
            return -1
    def model(self, x):
        # Here are matrix dot multiply get one value
        y = np.dot(x, self.w.T) + self.b
        # Use sign to nomalize the result
        predict_v = self.nomalize(y)
        return predict_v, y
    def update(self, x, y):
        # w = w + n*y_i*x_i
        self.w = self.w + self.eta*y*x
        # b = b + n*y_i
        self.b = self.b + self.eta*y
    def loss(slef, fx, y):
        return fx.astype(int)*y

    def train(self, count):
        update_count = 0
        while count > 0:
            # count--
            count = count - 1

            if len(self.train_data) <= 0:
                print("exception exit")
                break
            # random select one train data
            index = randint(0,len(self.train_data)-1)
            x = self.train_data[index]
            y = self.real_result.T[index]
            # wx+b
            predict_v, linear_y_v = self.model(x)
            # y_i*(wx+b) > 0, the classify is correct, else it's error
            if self.loss(y, linear_y_v) > 0:
                continue
            update_count = update_count + 1
            self.update(x, y)
        print("update count: ", update_count)
        pass
    def verify(self, verify_data, verify_result):
        size = len(verify_data)
        failed_count = 0
        if size <= 0:
            pass
        for i in range(size):
            x = verify_data[i]
            y = verify_result.T[i]
            if self.loss(y, self.model(x)[1]) > 0:
                continue
            failed_count = failed_count + 1
        success_rate = (1.0 - (float(failed_count)/size))*100
        print("Success Rate: ", success_rate, "%")
        print("All input: ", size, " failed_count: ", failed_count)

    def predict(self, predict_data):
        size = len(predict_data)
        result = []
        if size <= 0:
            pass
        for i in range(size):
            x = verify_data[i]
            y = verify_result.T[i]
            result.append(self.model(x)[0])
        return result



if __name__ == "__main__":
    # Init some parameters
    gradient = 2
    offset   = 10
    point_num = 1000
    train_num = 50000
    loader = TrainDataLoader()
    x, y, result, train_data =  loader.GenerateRandomData(point_num, gradient, offset)
    x_t, y_t, test_real_result, test_data =  loader.GenerateRandomData(100, gradient, offset)

    # First training
    perceptron = SimplePerceptron(train_data, result)
    perceptron.train(train_num)
    perceptron.verify(test_data, test_real_result)
    print("T1: w:", perceptron.w," b:", perceptron.b)

    # Draw the figure
    # 1. draw the (x,y) points
    plt.plot(x, y, "*", color='gray')
    plt.plot(x_t, y_t, "+")
    # 2. draw y=gradient*x+offset line
    plt.plot(x,x.dot(gradient)+offset, color="red")
    # 3. draw the line w_1*x_1 + w_2*x_2 + b = 0
    plt.plot(x, -(x.dot(float(perceptron.w.T[0]))+float(perceptron.b))/float(perceptron.w.T[1])
             , color='green')
    plt.show()

