# Gauss迭代法 输入系数矩阵mx、值矩阵mr、迭代次数n(以list模拟矩阵 行优先)
def Gauss(mx, mr, n=100):
    if len(mx) == len(mr):  # 若mx和mr长度相等则开始迭代 否则方程无解
        x = []  # 迭代初值 初始化为单行全0矩阵
        for i in range(len(mr)):
            x.append([0])
        count = 0  # 迭代次数计数
        while count < n:
            for i in range(len(x)):
                nxi = mr[i][0]
                for j in range(len(mx[i])):
                    if j != i:
                        nxi = nxi + (-mx[i][j]) * x[j][0]
                nxi = nxi / mx[i][i]
                x[i][0] = nxi
            count = count + 1
        return x
    else:
        return False


# 调用 Gauss(mx,mr,n=100) 示例


mx = [[8, -3, 2], [4, 11, -1], [6, 3, 12]]
mr = [[20], [33], [36]]
print(Gauss(mx, mr, 20))
