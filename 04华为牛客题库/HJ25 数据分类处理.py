# a=input().split()[1:]
# b = input()
b = '5 6 3 6 3 0'
b = set(b.split()[1:])
b = map(int,b)
b = sorted(b)
b=map(str, b)
b = list(b)
a = ['123', '456', '786', '453', '46', '7', '5', '3', '665', '453456', '745', '456', '786', '453', '123']
totalNum=0
res=""
for num in b:
    print('> ',num)
    singleRes,count="",0
    for i,v in enumerate(a):
        if num in v:
            singleRes+=str(i)+" "+v+" "
            totalNum+=2
            count+=1
    if count:
        singleRes=num+" "+str(count)+" "+singleRes
        totalNum+=2
    res+=singleRes
print((str(totalNum)+" "+res).rstrip())

