s = input()
#s = "I am a student"
res = ""
for c in s:
    if c.isalpha():
        res += c
    else:
        res += " "
print(*reversed(res.split()))

