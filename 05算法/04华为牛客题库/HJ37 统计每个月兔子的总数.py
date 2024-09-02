def Fibonacci(n):
    if n <= 2:
        return 1;
    else:
        return Fibonacci(n-1) + Fibonacci(n-2)

def main1():
    while True:
        try:
            input_n = input()
            print(Fibonacci(int(input_n)))
        except:
            break

if __name__ == '__main__':
    main1()