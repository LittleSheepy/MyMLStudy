

class A():
    # def __init__(self):
    #     print("A init")

    def A_func(self):
        print("A_Func_in")
        self.B_func()
        print("A_Func_out")


class B():
    def __init__(self):
        print("B init")

    def B_func(self):
        print("B_Func_in")
        #self.A_func()
        print("B_Func_out")


class C(A, B):
    def __init__(self):
        super(C,self).__init__()
        print("C init")

    # def B_func(self):
    #     print("C_Func")

if __name__ == '__main__':
    c = C()
    c.A_func()




