import ctypes
from ctypes import *


def load_dll(dll_path):
    return ctypes.cdll.LoadLibrary(dll_path)


def call_hello():
    dll.hello()


def call_func_type_base():
    dll.func_type_base.argtype = [c_float, c_float]
    dll.func_type_base.restype = c_float
    a = c_float(2.1)
    b = c_float(3.5)
    c = dll.func_type_base(a, b)
    print("python: out_c: ", c)


def call_func_arg_ptr():
    dll.func_arg_ptr.argtypes = [POINTER(c_int), POINTER(c_float)]
    dll.func_arg_ptr.restype = c_void_p

    int_a = c_int(0)
    float_b = c_float(0)

    print("python: byref(int_a):", byref(int_a))
    print("python: byref(float_b):", byref(float_b))
    dll.func_arg_ptr(byref(int_a), byref(float_b))      # byref()函数用于传输地址
    print("python: out_a:", int_a.value)
    print("python: out_b:", float_b.value)


# python2默认都是ASCII编码，python3中str类型默认是Unicode类型，
# 而ctypes参数需传入bytes-like object。因此python3中的字符串都需要转换编码
def call_func_arg_array():
    dll.func_arg_array.argtype = [c_char_p, POINTER(c_ubyte*16), c_char_p]
    dll.func_arg_array.restype = c_void_p

    # create_string_buffer函数会分配一段内存，产生一个c_char类型的字符串，并以NULL结尾
    # create_unicode_buffer函数，返回的是c_wchar类型
    str_info = create_string_buffer(b"Fine,thank you")

    # from_buffer_copy函数则是创建一个ctypes实例，并将source参数内容拷贝进去
    u_str_info = (c_ubyte*16).from_buffer_copy(b'0123456789abcdef')

    word = (c_char * 32)()          # c_char*32 是数组类型

    dll.func_arg_array(str_info, byref(u_str_info), word)
    print("python: str_info:", str_info.value, str_info.raw)
    for i in range(3):
        print("python: str_info: ", u_str_info[i], chr(u_str_info[i]))
    print("python: out_word:", word.value, word.raw)


class Rect(Structure):
    _fields_ = [
        ('index', c_int),
        ('info', c_char * 16)
    ]


def call_func_arg_struct():
    dll.func_arg_struct.argtypes = [Rect,  POINTER(Rect)]
    dll.func_arg_struct.restype = c_void_p

    rect_a = Rect(10, b"hello")
    rect_b = Rect(100, b"hello")
    dll.func_arg_struct(rect_a, byref(rect_b))
    print("python: rect_a:", rect_a.index, rect_a.info)
    print("python: rect_b:", rect_b.index, rect_b.info)

def call_func_arg_struct_array():
    dll.func_arg_struct_array.argtypes = [POINTER(Rect)]
    dll.func_arg_struct_array.restype = c_void_p

    rect_array = (Rect * 5)()
    for i in range(5):
        rect_array[i] = Rect(i, bytes("Hello_" + str(i), encoding='utf-8'))

    # 以下两种方法皆可
    print("第一种======参数=====rect_array")
    dll.func_arg_struct_array(rect_array)
    print("python: rect_b:", rect_array[0].index, rect_array[0].info)

    print("第一种======参数=====byref(rect_array[0])")
    dll.func_arg_struct_array(byref(rect_array[0]))
    print("python: rect_b:", rect_array[0].index, rect_array[0].info)

# 获取结构体列表
def call_func_res_struct_array():
    dll.func_res_struct_array.argtypes = [POINTER(c_int)]
    dll.func_res_struct_array.restype = POINTER(Rect)

    dll.freeRect.argtypes = [POINTER(Rect)]
    dll.freeRect.restype = c_void_p

    num = c_int(10)

    print("python: num:", num.value)
    rect_pt = dll.func_res_struct_array(byref(num))
    print("python: num:", num.value)

    # 结构体数组初始化
    # rect_pt.contents只能输出首元素的内容，rect_pt.contents.index
    rect_array = [rect_pt[i] for i in range(num.value)]

    for item in rect_array:
        print("python: index:", item.index, ", info:", item.info, item.info.decode('utf-8'))

    dll.freeRect(rect_pt)

class UserStruct(Structure):
    _fields_ = [
        ('user_id', c_long),
        ('name', c_char * 21)
    ]


class CompanyStruct(Structure):
    _fields_ = [
        ('com_id', c_long),
        ('name', c_char * 21),
        ('users', UserStruct * 100),
        ('count', c_int)
    ]


def c_print_company():
    library = dll
    library.Print_Company.argtypes = [POINTER(CompanyStruct)]
    library.Print_Company.restype = c_void_p

    user_array = (UserStruct * 100)()
    count = 2
    for i in range(count):
        user_array[i] = UserStruct(i, bytes("user_" + str(i), encoding='utf-8'))

    company = CompanyStruct(1, b"esunny", user_array, count)

    print("python: com_id, name, count:  ", company.com_id, company.name, company.count)
    for i in range(company.count):
        print("python: ", company.users[i].user_id, company.users[i].name)

    library.Print_Company(byref(company))

    print("python: com_id, name, count:  ", company.com_id, company.name, company.count)
    for i in range(company.count):
        print("python: ", company.users[i].user_id, company.users[i].name)


if __name__ == '__main__':
    dll_path = r"./PyCallDll/x64/Debug/PyCallDll.dll"
    dll = load_dll(dll_path)
    print("")
    c_print_company()













