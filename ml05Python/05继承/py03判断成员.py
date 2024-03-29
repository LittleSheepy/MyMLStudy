
class Base():
    def __init__(self):
        self.model_type = "det"


class Child1(Base):
    def __init__(self):
        super().__init__()
        self.model_type1 = "11"

class Child2(Base):
    def __init__(self):
        super().__init__()
        self.model_type2 = "22"


if __name__ == '__main__':
    print("")
    child1 = Child1()
    if hasattr(child1, 'model_type2'):
        print("child1 has model_type2")
    else:
        print("child1 does not have model_type2")
