num1 = ['zero','one','two','three','four','five','six',
       'seven','eight','nine','ten','eleven','twelve',
       'thirteen','fourteen','fifteen','sixteen',
       'seventeen','eighteen','nineteen']
num2 = [0,0,'twenty','thirty','forty','fifty','sixty',
       'seventy','eighty','ninety']

def hundred_less(n):
    if n < 20:
        return num1[n]
    ge = n % 10
    shi = (n // 10) % 10
    s = ""
    if shi > 0:
        s = num2[shi]
    if ge > 0:
        s = s + " " + num1[ge]
    return s

def hundred(n):
    ge = n % 10
    shi = (n // 10) % 10
    bai = (n // 100) % 10
    s = ""
    if bai > 0:
        s = num1[bai] + " hundred"
        if shi > 0 or ge > 0:
            s = s + " and"
            s = s + " " + hundred_less(shi*10 + ge)
    else:
        s = hundred_less(shi * 10 + ge)
    return s
def main():
    while True:
        try:
            s = ""
            input_long = int(input())

            n_million = input_long // 1000000
            n_thousand  = (input_long // 1000)%1000
            n_hundred = input_long % 1000
            if n_million > 0:
                s = s + hundred(n_million) + " million"
            if n_thousand > 0:
                if s != "":
                    s = s + " "
                s = s + hundred(n_thousand) + " thousand"
            if n_hundred > 0:
                if s != "":
                    s = s + " "
                s = s + hundred(n_hundred)
            print(s)
        except:
            break


if __name__ == '__main__':
    main()
