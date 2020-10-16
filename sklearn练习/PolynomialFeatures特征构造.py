import sklearn as sk
from sklearn.preprocessing import PolynomialFeatures


# 它是使用多项式的方法来进行的，如果有a，b两个特征，那么它的2次多项式为（1,a,b,a^2,ab, b^2），
# 这个多项式的形式是使用poly的效果。

"""
    def __init__(self, degree=2, interaction_only=False, include_bias=True,
                 order='C')
PolynomialFeatures有三个参数
degree：默认为2,控制多项式的度
interaction_only： 默认为False，如果指定为True，那么就不会有特征自己和自己结合的项，上面的二次项中没有a^2和b^2。
include_bias：默认为True。如果为True的话，那么就会有上面的 1那一项
"""
c = [[5, 10],[5,10]]
pl = PolynomialFeatures()

b = pl.fit_transform(c)

# [[  1.   5.  10.  25.  50. 100.]]
print(b)

pl = PolynomialFeatures(interaction_only=True)
b = pl.fit_transform(c)

# [[ 1.  5. 10. 50.]]
print(b)

pl = PolynomialFeatures(include_bias=False)
b = pl.fit_transform(c)

# [[ 1.  5. 10. 50.]]
print(b)


