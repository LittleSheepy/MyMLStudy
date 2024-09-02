def check(data, x, y):
    # xy
    for i in range(9):
        if (i != x) and data[y][i] == data[y][x]:
            return False
        if (i != y) and data[i][x] == data[y][x]:
            return False
    # grid
    xx = (x//3)*3
    yy = (y//3)*3
    for i in range(3):
        for j in range(3):
            if (xx+i == x) or(yy+j == y):
                continue
            if data[yy+j ][xx+i] == data[y][x]:
                return False
    return True
def func(data):
    for x in range(9):
        for y in range(9):
            if data[y][x] == 0:
                for i in range(9):
                    data[y][x] = i+1
                    if not check(data, x, y):
                        data[y][x] = 0
                        continue
                    if not func(data):
                        data[y][x] = 0
                        continue
                    else:
                        return True
                return False
    return True
def main():
    while True:
        try:
            sudu_data = []
            for i in  range(9):
                input_s = list(map(int, input().split(" ")))
                sudu_data.append(input_s)
            func(sudu_data)

            for i in range(9):
                sudu_data[i] = list(map(str, sudu_data[i]))
                print(' '.join(sudu_data[i]))
        except:
            break


if __name__ == '__main__':
    print("")
    main()
