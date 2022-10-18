s = input()
while len(s):
    s_new = s[:8]
    s = s[8:]
    s_out = s_new.ljust(8, "0")
    print(s_out)







