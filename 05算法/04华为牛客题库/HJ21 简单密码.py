s = input()
res = []
for c in s:
    if c in 'abc':
        res.append('2')
    elif c in 'def':
        res.append('3')
    elif c in 'ghi':
        res.append('4')
    elif c in 'jkl':
        res.append('5')
    elif c in 'mno':
        res.append('6')
    elif c in 'pqrs':
        res.append('7')
    elif c in 'tuv':
        res.append('8')
    elif c in 'wxyz':
        res.append('9')
    elif c == 'Z':
        res.append('a')
    elif c.isupper():
        res.append(chr(ord(c.lower())+1))
    else:
        res.append(c)
print(''.join(res))