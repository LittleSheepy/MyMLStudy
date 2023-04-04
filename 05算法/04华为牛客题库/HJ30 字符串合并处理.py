# 输入
#str1,str2 = input().split(" ")
str1 = "Gec"
str2 = "fab"

# 合并
str_all = str1 + str2
str_all = list(str_all)
# 排序
str_all[::2] = sorted(str_all[::2])
str_all[1::2] = sorted(str_all[1::2])

# 转换
for c in str_all:
    try:
        c_int = int(c,16)
        c_bin = bin(c_int)[2:]
        c_bin_str = str(c_bin)
        c_bin_str = c_bin_str.rjust(4,"0")
        c_bin_str = c_bin_str[::-1]
        #c_bin = bin(c_bin_str)
        c_int = int(c_bin_str, 2)
        c_hex = hex(c_int)[2:].upper()
        print(c_hex,end="")
    except:
        print(c, end="")