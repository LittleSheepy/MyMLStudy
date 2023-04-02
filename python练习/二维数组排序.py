import numpy as np
np.random.seed(0)
# 生成随机二维数组
arr = np.random.randint(0, 10, (2, 3))
print("原始数组：\n{}".format(arr))
# 将数组转换为一维，记录原始位置
index = np.arange(arr.size)                 # [ 0  1  2  3  4  5  6  7  8  9 10 11]
arr1d = arr.ravel()                         # [5 0 3 3 7 9 3 5 2 4 7 6]
# 根据值对一维数组进行排序
sort_index = np.argsort(arr1d)              # [ 1  8  2  3  6  9  0  7 11  4 10  5]
# 通过排序索引重新排序一维数组，并返回原始位置
# sort_index_index = np.argsort(sort_index)   # [ 6  0  2  3  9 11  4  7  1  5 10  8]
sorted_index = np.zeros_like(sort_index)
sorted_index[sort_index] = index            # [ 6  0  2  3  9 11  4  7  1  5 10  8]
# 将一维的index 转换为 二维的index
sorted_arr = np.unravel_index(sorted_index, arr.shape)
# 将二维位置作为元组列表返回
result = [(sorted_arr[0][i], sorted_arr[1][i]) for i in range(len(index))]
print("排序后数组：\n{}".format(arr[sorted_arr]))
print("排序后位置列表：\n{}".format(result))