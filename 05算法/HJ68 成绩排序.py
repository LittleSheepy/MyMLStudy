def main():
    while True:
        try:
            input_num = int(input())
            input_flg = input()
            flg = False
            if int(input_flg) == 0:
                flg = True
            ls = []
            for i in range(input_num):
                name,score = input().split()
                ls.append([name, int(score)])
            ls.sort(key=lambda x:x[1],reverse=flg)
            for x in ls:
                print(*x)
        except:
            break


if __name__ == '__main__':
    main()
