str_list = input().split(' ')
# for i in range(len(str_list)-1, -1, -1):
#     print(str_list[i], end=" ")
str_list = str_list[::-1]
for s in str_list:
    print(s, end=" ")