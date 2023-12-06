import ctypes
from ctypes import *


def load_dll(dll_path):
    return ctypes.cdll.LoadLibrary(dll_path)


def call_hello():
    dll.hello()


def call_func_arg_base():
    dll.addfloat.argtype = [c_float, c_float]
    dll.addfloat.restype = c_float
    a = c_float(2.1)
    b = c_float(3.5)
    c = dll.func_arg_base(a, b)
    print(c)


if __name__ == '__main__':
    dll_path = r"./PyCallDll/x64/Debug/PyCallDll.dll"
    dll = load_dll(dll_path)
    call_func_arg_base()













