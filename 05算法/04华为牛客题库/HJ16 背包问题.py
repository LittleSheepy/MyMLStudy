# 空间10  件数5
n, m = 10, 5
w = [0, 1, 2, 3, 4, 5]
v = [0, 1, 2, 3, 4, 5]
dp = [[0]*(n+1) for _ in range(m+1)]
for i in range(1,m+1):   # 件数
    for j in range(1,n+1):  # 空间
        if j-w[i]>=0:
            dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]]+v[i])
        else:
            dp[i][j] = dp[i-1][j]
print(dp[m][n])