import sys

for line in sys.stdin:
    a = line.split()
    print(int(len(a[-1])))
