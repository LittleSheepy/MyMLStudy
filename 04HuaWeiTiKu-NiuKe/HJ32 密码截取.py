def test2():
    s = "abbba"
    n = len(s)
    single_center, double_center = [1]*n, [0]*n     # [1, 1, 1, 1, 1, 1, 1]  [0, 0, 0, 0, 0, 0, 0]
    for i in range(1,n):
        j = 0
        while j < min(i, n-i-1):
            j += 1
            if s[i-j]==s[i+j]:
                single_center[i] += 2
            else:
                break
    for i in range(0,n-1):
        j = 0
        while j <= min(i, n-i-2):
            if s[i-j] == s[i+1+j]:
                double_center[i] += 2
            else:
                break
            j += 1

    print(max(max(single_center), max(double_center)))

def max_len(string):
    abba = []
    aba = []
    n = len(string)
    #遍历寻找对称位置
    for i in range(n-1):
        current = i
        next_one = i+1
        if string[current-1] == string[next_one]:
            aba.append(i)
        if string[current] == string[next_one]:
            abba.append(i)
    length = []
    #遍历对称位置寻找最长对称字符串
    for j in abba:
        first = j
        last = j+1
        while first>=0 and last<len(string) and string[first]==string[last]:
            first+=-1
            last+=1
            #CABBA，第一循环时，符合条件的只有2个字符，而此时last-first=3,所以需要减去1
            length.append(last-first-1)
    for k in aba:
        first = k-1
        last = k+1
        while first>=0 and last<len(string) and string[first] == string[last]:
            first+=-1
            last +=1
            length.append(last-first-1)
    if len(length)==0:
        return 0
    else:
        return max(length)

while True:
    try:
        string = "abbaaab"
        print(max_len(string))
    except:
        break

# def test1():
#     str = input()
#     # str = "abbaaab"
#     n = len(str)
#     list = []
#     for i in range(0, n - 1):
#         for j in range(1, n):
#             # if str[j] == str[i] and str[i+1:j] == str[j-1:i:-1]:
#             if str[i:j] == str[j:i:-1]:
#                 list.append(len(str[i:j + 1]))
#     print(max(list))










