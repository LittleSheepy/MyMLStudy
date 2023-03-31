cnt = int(input())
dic = {}
for i in range(cnt):
    k, v = input().split(" ")
    k = int(k)
    v = int(v)
    if k in dic:
        dic.update({k: dic[k]+v})
    else:
        dic.update({k: v})
k_list = sorted(dic, key=lambda x:x)
for k in k_list:
    print(k, dic[k])



