ipClass2num = {
    'A':0,
    'B':0,
    'C':0,
    'D':0,
    'E':0,
    'ERROR':0,
    'PRIVATE':0,
}
# 私有IP地址和A,B,C,D,E类地址是不冲突的，也就是说需要同时+1
def check_ip(ip:str):
    ip_bit = ip.split('.')
    if len(ip_bit) != 4 or '' in ip_bit:  #ip 的长度为4 且每一位不为空
        return False
    for i in ip_bit:
        if int(i)<0 or int(i) >255:   #检查Ip每一个10位的值范围为0~255
            return False
    return True
def check_mask(mask:str):
    if not check_ip(mask):
        return False
    if mask == '255.255.255.255' or mask == '0.0.0.0':
        return False
    mask_list = mask.split('.')
    if len(mask_list) != 4:
        return False
    mask_bit = []
    for i in mask_list:#小数点隔开的每一数字段
        i = bin(int(i))#每一数字段转换为每一段的二进制数字段
        i = i[2:] #从每一段的二进制数字段的第三个数开始，因为前面有两个ob
        mask_bit.append(i.zfill(8)) #.zfill:返回指定长度的字符串，原字符串右对齐，前面填充0
    whole_mask = ''.join(mask_bit)
#     print(whole_mask)
    whole0_find = whole_mask.find("0")#查0从哪里开始
    whole1_rfind = whole_mask.rfind("1")#查1在哪里结束
    if whole1_rfind+1 == whole0_find:#两者位置差1位为正确
        return True
    else:
        return False
def check_private_ip(ip:str):
    # 三类私网
    ip_list = ip.split('.')
    if ip_list[0] == '10': return True
    if ip_list[0] == '172' and 16<=int(ip_list[1])<=31: return True
    if ip_list[0] == '192' and ip_list[1] == '168': return True
    return False
while True:
    try:
        s = input()
        ip = s.split('~')[0]
        mask = s.split('~')[1]
        if check_ip(ip):
            first = int(ip.split('.')[0])
            if first==127 or first==0:
                # 若不这样写，当类似于【0.*.*.*】和【127.*.*.*】的IP地址的子网掩码错误时，会额外计数
                continue
            if check_mask(mask):
                if check_private_ip(ip):
                    ipClass2num['PRIVATE'] += 1
                if 0<first<127:
                    ipClass2num['A'] += 1
                elif 127<first<=191:
                    ipClass2num['B'] += 1
                elif 192<=first<=223:
                    ipClass2num['C'] += 1
                elif 224<=first<=239:
                    ipClass2num['D'] += 1
                elif 240<=first<=255:
                    ipClass2num['E'] += 1
            else:
                ipClass2num['ERROR'] += 1
        else:
            ipClass2num['ERROR'] += 1
    except:
        break
for v in ipClass2num.values():
    print(v, end=' ')
