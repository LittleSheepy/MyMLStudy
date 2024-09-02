def int2bin(lst):
    res = ''
    for i in lst:
        a = bin(i)[2:]
        b = '0' * (8 - len(a)) + str(a)
        res += b
    return res
def check_ip(ip_lst):
    if len(ip_lst) != 4 or '' in ip_lst:
        return False
    for i in ip_lst:
        if not 0 <= i <= 255:
            return False
    return True


def check_mask(mask):
    if not check_ip(mask):
        return False
    res = int2bin(mask)
    if res.find('0') == res.rfind('1') + 1:
        return True
    return False
def func(ip1_list, ip2_list, mask_list):
    result = 0
    for i in range(4):
        if ip1_list[i]&mask_list[i] != ip2_list[i]&mask_list[i]:
            result = 2
            break
    return result

def panduan(ip1_list):
    for i in range(4):
        if ip1_list[i] < 0 or ip1_list[i] > 255:
            return False
    return True

def main():
    while True:
        try:
            mask_list = [int(s) for s in input().split(".")]
            ip1_list = [int(s) for s in input().split(".")]
            ip2_list = [int(s) for s in input().split(".")]
            if not (panduan(mask_list) and panduan(ip1_list) and panduan(ip2_list) and check_mask(mask_list)):
                print(1)
                break
            print(func(ip1_list, ip2_list, mask_list))
        except:
            break



if __name__ == '__main__':
    main()