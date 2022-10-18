i_str = input()
i_str = i_str[::-1]
s = set()
for i in i_str:
    if i not in s:
        print(i, end='')
    s.add(i)