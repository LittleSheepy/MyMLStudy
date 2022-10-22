#二分查找，其中ls是从左到右升序
def find_pos(ls, num,start = 0, end = None):
    if end == None:
        end = len(ls)
    while start < end:
        index = (start+end)//2
        if ls[index]<num:
            start = index+1
        else:
            end = index
    return start

def dplist(raw_list):
    dp = [1]*len(raw_list)
    stu_list = [raw_list[0]]
    for i in range(1,len(raw_list)):
        if raw_list[i] > stu_list[-1]:
            stu_list.append(raw_list[i])
            dp[i] = len(stu_list)
        else:
            index = find_pos(stu_list,raw_list[i])
            stu_list[index] = raw_list[i]
            dp[i] = index+1
    return dp


# num =int(input())
# ls = list(map(int,input().split()))

num = 10
#ls = [186, 186, 150, 200, 160, 130, 197, 200]
#ls = [186, 186, 150, 200, 160, 145, 130, 150, 197, 200]
ls = [11, 21, 31, 41, 51, 2, 4, 11, 5, 200]

l_ls = dplist(ls)       # [1, 1, 1, 2, 2, 1, 3, 4]
r_ls = dplist(ls[::-1])[::-1]
total_ls = [l_ls[i]+r_ls[i]-1 for i in range(len(ls))]
print(num-max(total_ls))

