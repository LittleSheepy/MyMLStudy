while True:
    try:
        list_c=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'] # 初始字母表
        list_b=list(input())                            # 关键字
        list_d=sorted(set(list_b),key=list_b.index)     # 关键字剔除重复项
        list_out1=list(input())                         # 需被翻译项
        list_out2=''                                    # 翻译结果
        for i in range(len(list_c)):
            if list_c[i] not in list_d:
                list_d+=list_c[i]                   #制作密码本
        for i in range(len(list_out1)):
            for j in range(len(list_c)):
                if list_c[j] in list_out1[i]:
                    list_out2+=list_d[j]#               嵌套循环加密加密
        print(list_out2)#                         输出结果
    except EOFError as e:
        break


