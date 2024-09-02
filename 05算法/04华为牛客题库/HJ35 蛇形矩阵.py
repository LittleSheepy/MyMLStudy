"""
1 3 6 10 15

  2 5  9 14

    4  8 13

       7 12

         11
"""
n = int(input().strip())
for i in range(1,n+1) :
    for j in range(i,n+1) :
        # 输出的公式是转化后的，其实也可以写去括号前，int()的作用是将结果转化成整型，因为计算结果是有小数的，end = ' '作用是将内层循环的计算结果以空格隔开
        out = int(((j+j**2)/2)-i+1)
        print(out, end = ' ')
    # 一次循环结束后打印空，用作换行
    print()