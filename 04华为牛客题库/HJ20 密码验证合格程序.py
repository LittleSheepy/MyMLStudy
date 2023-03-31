def checkLegal(pswd):
    if len(pswd) <= 8:return False
    else:
        #最大重复子串长度2+
        sub = []
        for i in range(len(pswd)-2):
            sub.append(pswd[i:i+3])
        if len(set(sub)) < len(sub):return False
        #check type
        type_ = 0
        import re
        Upper = '[A-Z]'
        Lowwer = '[a-z]'
        num = '\d'
        chars = '[^A-Za-z0-9_]'
        patterns = [Upper, Lowwer, num, chars]
        for pattern in patterns:
            pw = re.search(pattern, pswd)
            if pw : type_ += 1
        return True if type_ >= 3 else False
while True:
    try:
        pswd = input()
        print('OK' if checkLegal(pswd) else 'NG')
    except:
        break