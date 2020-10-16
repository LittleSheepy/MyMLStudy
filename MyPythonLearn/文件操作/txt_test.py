#encoding=utf8
import io

f=open("text_test.txt",'w')
f.write('TXT例子1\nTXT例子2\n')
f.close()
f=open("text_test.txt",'r+')
#print(f.read()) # 先读一次，会添加到文件末尾。 "r+"换成"a"就不用先读取一次了
f.write('TXT例子3\nTXT例子4\n')
f.write('TXT例子5\nTXT例子6\n')
f.close()

f=open("text_test.txt")
print(f.read())
f.close()
print("*"*10)
with open("text_test.txt") as f:
    for line in f:
        print(line)
        
print("*"*10)

with open("text_test.txt") as f:
    i=0
    while True:
        line = f.readline()
        if not line:
            break
        print(line)
        i+=1
    print(i)
