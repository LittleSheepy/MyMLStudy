a = 20
str = input()
str_set = set(str)
for c in str_set:
    if str.count(c) <= a:
        a = str.count(c)
for c in str_set:
    if str.count(c) == a:
        str = str.replace(c, '')
print(str)