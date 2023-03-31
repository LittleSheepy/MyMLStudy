n = int(input())
for i in range(2, int(n**0.5)+1):
    if n == 1:break
    while n % i == 0:
        print(i, end=' ')
        n = n // i
if n > 2:
    print(n)
