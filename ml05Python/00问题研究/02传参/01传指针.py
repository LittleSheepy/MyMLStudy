

def para_list(p_list:list):
    p_list.append(1)


if __name__ == '__main__':
    print("")
    my_list = []
    para_list(my_list)
    print(my_list)