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

    word = (c_char * 32)()

    dll.func_arg_array(str_info, byref(u_str_info), word)
    print("python: str_info:", str_info.value, str_info.raw)
    for i in range(3):
        print("python: str_info: ", u_str_info[i], chr(u_str_info[i]))
    print("python: out_word:", word.value, word.raw)

if __name__ == '__main__':
    dll_path = r"./PyCallDll/x64/Debug/PyCallDll.dll"
    dll = load_dll(dll_path)
    print("")
    call_func_arg_array()













