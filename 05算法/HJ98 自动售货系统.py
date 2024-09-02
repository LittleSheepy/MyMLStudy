
class good():
    def __init__(self):
        self.name = ""
        self.price = 0
        self.num = 0

class good_sys():
    def __init__(self):
        self.good_list = {}
        self.good_names = ["A1", "A2", "A3", "A4", "A5", "A6"]
        self.money_num = {"1":0, "2":0, "5":0, "10":0}
        self.money_index = ["1", "2", "5", "10"]
        self.good_prices = {"A1":2, "A2":3, "A3":4, "A4":5, "A5":8, "A6":6}
        for name in self.good_names:
            g = good()
            g.name = name
            g.price = self.good_prices[name]
            self.good_list[name] = g
        self.yu_e = 0

    def have_no_good(self):
        for good in self.good_list:
            if self.good_list[good].num > 0:
                return False
        return True

#       AN 数量
# r 22-18-21-21-7-20 3-23-10-6;c;q0;p 1;b A6;c;b A5;b A1;c;q1;p 5;
def main():
    while True:
        try:
            gs = good_sys()
            input_s = input()
            # input_s = "r 22-18-21-21-7-20 3-23-10-6;c;q0;p 1;b A6;c;b A5;b A1;c;q1;p 5;"
            # input_s = "r 26-3-26-2-14-10 2-1-15-18;p 5;c;c;p 2;c;b A4;c;q1;q0;p 2;b A4;p 5;q0;c;q0;q1;q0;c;c;p 10;p 1;q0;"
            cmds = input_s.split(";")
            for cmd in cmds:
                if cmd[0] == "r":
                    cmd_tmps = cmd.split()
                    goods_num = list(map(int, cmd_tmps[1].split("-")))
                    for i in range(6):
                        gs.good_list[gs.good_names[i]].num = goods_num[i]

                    moneys_num = list(map(int, cmd_tmps[2].split("-")))
                    for i in range(4):
                        gs.money_num[gs.money_index[i]] = moneys_num[i]
                    print("S001:Initialization is successful")
                elif cmd[0] == "c":
                    if gs.yu_e == 0:
                        print("E009:Work failure")
                    else:
                        w10, w5, w2, w1 = 0, 0, 0, 0  # 记录已经找出的零钱
                        pq = gs.yu_e
                        dic_q = gs.money_num
                        add = 0
                        while pq > 0:  # 循环直到找零完成
                            if pq >= 10 and dic_q['10'] >= 1:  # 可以找10元时
                                pq -= 10  # 余额减10
                                w10 += 1  # 已经找出的零钱+1
                                dic_q['10'] -= 1  # 零钱10数量-1
                            elif pq >= 5 and dic_q['5'] >= 1:  # 可以找5元时
                                pq -= 5
                                w5 += 1
                                dic_q['5'] -= 1
                            elif pq >= 2 and dic_q['2'] >= 1:
                                pq -= 2
                                w2 += 1
                                dic_q['2'] -= 1
                            elif pq >= 1 and dic_q['1'] >= 1:
                                pq -= 1
                                w1 += 1
                                dic_q['1'] -= 1
                            else:
                                pq -= 1  # 耍赖，如果因零钱不足导致不能退币，则尽最大可能退币，以减少用户损失。
                                add = add + 1
                        print('1 yuan coin number={}'.format(w1))
                        print('2 yuan coin number={}'.format(w2))
                        print('5 yuan coin number={}'.format(w5))
                        print('10 yuan coin number={}'.format(w10))
                        gs.yu_e = add
                elif cmd[0] == "q":
                    if not cmd in ["q 0", "q 1"]:
                        print("E010:Parameter error")
                        continue
                    if cmd[2] == "0":
                        for good_name in gs.good_names:
                            print(" ".join([gs.good_list[good_name].name,str(gs.good_list[good_name].price),str(gs.good_list[good_name].num)]) )

                    elif cmd[2] == "1":
                        for money_name in gs.money_index:
                            print(money_name," yuan coin number=", gs.money_num[money_name])
                elif cmd[0] == "b": # 购买商品
                    cmd_tmps = cmd.split()
                    good_name = cmd_tmps[1]
                    if not good_name in gs.good_names:
                        print("E006:Goods does not exist")
                        continue
                    if gs.good_list[good_name].num == 0:
                        print("E007:The goods sold out")
                        continue
                    if gs.good_list[good_name].price > gs.yu_e:
                        print("E008:Lack of balance")
                        continue
                    gs.yu_e = gs.yu_e - gs.good_list[good_name].price
                    print("S003:Buy success,balance="+str(gs.yu_e))
                elif cmd[0] == "p": # 投币
                    cmd_tmps = cmd.split()
                    money = cmd_tmps[1]
                    dic_q = gs.money_num
                    if not money in gs.money_index:
                        print("E002:Denomination error")
                        continue
                    if money not in ["1", "2"] and int(money) >= (dic_q['1'] + dic_q['2'] * 2):  # 存钱盒中1元和2元面额钱币总额小于本次投入的钱币面额
                        print('E003:Change is not enough, pay fail')
                        continue
                    if gs.have_no_good():
                        print("E005:All the goods sold out")
                        continue
                    gs.yu_e = gs.yu_e + int(money)
                    gs.money_num[money] = gs.money_num[money] + 1
                    print("S002:Pay success,balance="+str(gs.yu_e))
            break
        except:
            break


if __name__ == '__main__':
    # print("")
    main()

"""
r 22-18-21-21-7-20 3-23-10-6;c;q0;p 1;b A6;
c;
b A5;b A1;c;q1;p 5;

S001:Initialization is successful
E009:Work failure
E010:Parameter error
S002:Pay success,balance=1
E008:Lack of balance

1 yuan coin number=1
2 yuan coin number=0
5 yuan coin number=0
10 yuan coin number=0

E008:Lack of balance
E008:Lack of balance
E009:Work failure
E010:Parameter error
S002:Pay success,balance=5

"""