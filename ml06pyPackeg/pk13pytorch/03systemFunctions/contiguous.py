import torch
t = torch.arange(1,13).reshape(3,4)
print(t)
""" t
tensor([[ 1,  2,  3,  4],
        [ 5,  6,  7,  8],
        [ 9, 10, 11, 12]])
"""
# t.stride() = (4,1)每隔4个元素到下一行，每隔1个元素到下一列
print(t.stride())   # (4, 1)
# 将行和列交换位置，transpose
t2 = t.transpose(1,0)
print(t2)
""" t2
tensor([[ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11],
        [ 4,  8, 12]])
"""
# t2.stride() = (1,4)每隔1个元素到下一行，每隔4个元素到下一列（在原来t的基础上）
print(t2.stride())      # (1, 4)
print(t.is_contiguous())    # True
print(t2.is_contiguous())   # False
# 无论是t还是t2还是t3，摊平展开之后的结果都是一样的，flatten
print(t.flatten())
#t tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])

print(t2.flatten())
#t2 tensor([ 1,  5,  9,  2,  6, 10,  3,  7, 11,  4,  8, 12])

t3 = t2.contiguous()
print(t3.stride())      # (3, 1)
print(t3.flatten())
#t3 tensor([ 1,  5,  9,  2,  6, 10,  3,  7, 11,  4,  8, 12])
# 非连续位置存储的元素在进行view操作的时候会报错，t2.view(12,1)报错
try:
    t2.view(12,1)
except Exception as e:
    print(e)

t12_1 = t.view(12,1)
print(t.view(12,1))
"""
tensor([[ 1], [ 2], [ 3], [ 4], [ 5], [ 6], [ 7], [ 8], [ 9], [10], [11], [12]])
"""
# t2.contiguous()变为连续顺序存储，结果就不会报错，可以进行view操作
t312_1 = t3.view(12,1)
print(t312_1)
pass
"""
tensor([[ 1], [ 5], [ 9], [ 2], [ 6], [10], [ 3], [ 7], [11], [ 4], [ 8], [12]])
"""


