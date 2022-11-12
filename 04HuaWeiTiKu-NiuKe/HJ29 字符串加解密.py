def check(a,b):
    L1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    L2 = "BCDEFGHIJKLMNOPQRSTUVWXYZAbcdefghijklmnopqrstuvwxyza1234567890"
    result = ""
    if b == 1:
        for i in a:
            result += L2[L1.index(i)]
    elif b == -1:
        for i in a:
            result += L1[L2.index(i)]
    return result
while True:
    try:
        print(check(input(),1))
        print(check(input(), -1))

    except:
        break
