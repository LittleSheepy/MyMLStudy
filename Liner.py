import math
import numpy as np
from scipy import stats

testX = [174.5, 171.2, 172.9, 161.6, 123.6, 112.1, 107.1, 98.6, 98.7, 97.5, 95.8, 93.5, 91.1, 85.2, 75.6, 72.7, 68.6,
         69.1, 63.8, 60.1, 65.2, 71, 75.8, 77.8]
testY = [88.3, 87.1, 88.7, 85.8, 89.4, 88, 83.7, 73.2, 71.6, 71, 71.2, 70.5, 69.2, 65.1, 54.8, 56.7, 62, 68.2, 71.1,
         76.1, 79.8, 80.9, 83.7, 85.8]


def computeCorrelation(X, Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0, len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2

    SST = math.sqrt(varX * varY)
    print("使用math库：r：", SSR / SST, "r-squared：", (SSR / SST) ** 2)
    return


computeCorrelation(testX, testY)

x = np.array(testX)
y = np.array(testY)
# 拟合 y = ax + b
poly = np.polyfit(x, y, deg=1)
print("使用numpy库：a：" + str(poly[0]) + "，b：" + str(poly[1]))


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # a、b、r
    print("使用scipy库：a：", slope, "b：", intercept, "r：", r_value, "r-squared：", r_value ** 2)


rsquared(testX, testY)