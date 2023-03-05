#a = input()
#b = input()
# a = "10.0.3.193"
# a = list(map(int,a.split(".")))
# out_a = 0
# for i,n in enumerate(a[::-1]):
#     out_a += 256**i * n
print(sum([256**i*j for i,j in enumerate(list(map(int,input().split(".")))[::-1])]))


b = int(input())
print('.'.join([str((b) // (256 ** i) % 256) for i in range(3, -1, -1)]))