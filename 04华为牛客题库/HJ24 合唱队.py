def find_len(b):
    nums = b
    if not nums:
        return 0
    dp = [1] * len(nums)  # 先初始化dp值，初始全设为1，初始只有自己
    for i in range(1, len(nums)):  # 从初始的右边的人开始
        for j in range(i):  # 从初始（第0个）到第j个人的左边
            if (
                nums[i] > nums[j] and dp[i] < dp[j] + 1
            ):  # 若是第i个人的身高大于第j个人，而且自己的左边是j的话[dp[j] + 1]，得到的dp值大于自己本来得到的自己左边可以有多少人的计算[dp[i]]
                dp[i] = dp[j] + 1  # 那么i就加入j的排队序列
    return dp


while True:
    try:
        student_num = int(input())  # 输入需要排队的学生数
        student_tall = input()  # 输入所有的学生身高
        student_tall = student_tall.split()
        for i in range(student_num):
            student_tall[i] = int(student_tall[i])  # 将输入的字符转化为整数

        # student_num = 8
        # student_tall = [186, 186, 150, 200, 160, 130, 197, 200]
        len_all = []
        #len_1 = find_len(student_tall)  # 获得正向（从小到大排序的dp），dp值即是每个学生的左边最多有几个人（包括自己）
        nums = student_tall
        dp = [1] * len(nums)  # 先初始化dp值，初始全设为1，初始只有自己
        for i in range(1, len(nums)):  # 从初始的右边的人开始
            for j in range(i):  # 从初始（第0个）到第j个人的左边
                if (
                        nums[i] > nums[j] and dp[i] < dp[j] + 1
                ):  # 若是第i个人的身高大于第j个人，而且自己的左边是j的话[dp[j] + 1]，得到的dp值大于自己本来得到的自己左边可以有多少人的计算[dp[i]]
                    dp[i] = dp[j] + 1  # 那么i就加入j的排队序列
        len_1 = dp


        len_2 = find_len(
            student_tall[::-1]
        )  # 获得逆向从小到大（从大到小）len_2[::-1]代表每个学生的右边最多有几个人（包括自己）


        # nums = student_tall[::-1]
        # dp = [1] * len(nums)  # 先初始化dp值，初始全设为1，初始只有自己
        # for i in range(1, len(nums)):  # 从初始的右边的人开始
        #     for j in range(i):  # 从初始（第0个）到第j个人的左边
        #         if (
        #                 nums[i] > nums[j] and dp[i] < dp[j] + 1
        #         ):  # 若是第i个人的身高大于第j个人，而且自己的左边是j的话[dp[j] + 1]，得到的dp值大于自己本来得到的自己左边可以有多少人的计算[dp[i]]
        #             dp[i] = dp[j] + 1  # 那么i就加入j的排队序列
        # len_2 = dp



        len_len_1 = len(len_1)
        len_3 = []
        for i in range(len_len_1):
            len_3.append(len_1[i] + len_2[::-1][i])  # 两个相加就是每个学生的左右最多有几个人，结果包含两个自己
        print(
            student_num - max(len_3) + 1
        )  # 按照输出要求，要得到踢出去几个人，所以需要len_3 -1 就是每个学生左右加上自己最多有几个人，取最大的那个学生的值，再用总数减去就是踢出去最少的人
    except:
        break
