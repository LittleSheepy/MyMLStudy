import ctypes
from ctypes import *

# 基础类型
def type_base():
    # one character bytes, bytearray or integer
    char_type1 = c_char(b'A')
    char_type2 = c_char(bytearray([65]))
    char_type3 = c_char(2)
    print(char_type1, char_type1.value)
    print(char_type2, char_type2.value)
    print(char_type3, char_type3.value)

    string_type = c_wchar_p("abc")
    print(string_type, string_type.value)

    int_type = c_int(2)
    print(int_type, int_type.value)

def type_array():
    char_array = c_char * 3
    char_array_obj = char_array(1, b"2", b'a')
    print(char_array_obj, char_array_obj.value)

    int_array = (c_int * 3)(1, 2, 3)
    for i in int_array:
        print(i)

    char_array_2 = (c_char * 3)(1, 2, 3)
    print(char_array_2.value)   # 通过value方法获取值只适用于字符数组


# 指针类型
def type_ptr():
    #
    # pointer()用于将对象转化为指针
    int_obj = c_int(3)
    int_p = pointer(int_obj)
    print(int_p)
    # 使用contents方法访问指针
    print(int_p.contents)
    # 获取指针指向的值
    print(int_p[0])
    print(int_p.contents.value)

    # POINTER()用于定义某个类型的指针
    int_p = POINTER(c_int)
    # 实例化
    int_obj = c_int(4)
    int_p_obj = int_p(int_obj)
    print(int_p_obj)
    print(int_p_obj.contents)
    print(int_p_obj[0])

    # 创建空指针的方式
    null_ptr = POINTER(c_int)()
    print(bool(null_ptr))

    # 指针类型的转换
    int_p = pointer(c_int(4))
    print(int_p)

    char_p_type = POINTER(c_char)
    print(char_p_type)

    cast_type = cast(int_p, char_p_type)
    print(cast_type)

def type_struct():
    pass

















if __name__ == '__main__':
    print("")
    type_ptr()

