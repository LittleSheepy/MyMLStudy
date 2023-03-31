"""
题解1：字典存储，循环比较
"""
# N, m = map(int, input().split())
# main_parts, accessories = {}, {}
# for i in range(1, m + 1):
#     v, p, q = map(int, input().split())
#     if q == 0:
#         main_parts[i] = [v, v*p]
#     else:
#         if q in accessories:
#             accessories[q].append([v, v*p])
#         else:
#             accessories[q] = [[v, v*p]]

N, m = 5, 5
main_parts = {3: [1, 3], 4: [1, 2], 5: [1, 1]}
accessories = {5: [[2, 6], [2, 6]]}
dp = [0] * (N + 1)

for key, value in main_parts.items():
    w, v = [], []
    w.append(value[0])  # 1、主件
    v.append(value[1])
    if key in accessories:  # 存在附件
        w.append(w[0] + accessories[key][0][0])  # 2、主件+附件1
        v.append(v[0] + accessories[key][0][1])
        if len(accessories[key]) > 1:  # 附件个数为2
            w.append(w[0] + accessories[key][1][0])  # 3、主件+附件2
            v.append(v[0] + accessories[key][1][1])
            w.append(w[0] + accessories[key][0][0] + accessories[key][1][0])  # 4、主件+附件1+附件2
            v.append(v[0] + accessories[key][0][1] + accessories[key][1][1])
    for j in range(N, -1, -1):  # 物品的价格是10的整数倍
        for k, _ in enumerate(w):
            if j - w[k] >= 0:
                dp[j] = max(dp[j], dp[j - w[k]] + v[k])
print(dp[N])
