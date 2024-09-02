
def main_my():
    while True:
        try:
            input_n = float(input())
            h = input_n
            sum = 0
            last = 0
            for i in range(5):
                sum = sum + h * 2
                h = h / 2
                last = h
                # print("     ",sum)
                # print("     ",last)
            print(float(sum-input_n))
            print(last)
        except:
            break

if __name__ == '__main__':
    main_my()