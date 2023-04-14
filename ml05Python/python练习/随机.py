import random

# 随机生成整数
num = random.randint(1, 50) # 闭区间
print(num)

# 随机生成多个不同的数
nums = random.sample(range(0, 10), 8)
print(nums)

# 随机偶数
num = random.randrange(0, 101, 2) # 左闭右开区间
print(num)

# 随机生成多个字符
alphabet = '0123456789'
characters = random.sample(alphabet, 5)
print(characters)







