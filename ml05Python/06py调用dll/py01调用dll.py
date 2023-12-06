import ctypes

class CKDAI_Infer(ctypes.Structure):
    _fields_ = [('m_pEngine', ctypes.c_void_p), ('m_tTimer', ctypes.c_float)]

# 加载 DLL
my_dll = ctypes.cdll.LoadLibrary(r"D:\04Bin\KDAI_Infer.dll")

infer = CKDAI_Infer()
my_dll.createInfer.restype = ctypes.c_void_p
aa = my_dll.createInfer()
my_dll.Infer_test1.argtypes = [ctypes.c_void_p]
my_dll.Infer_test1.restype = ctypes.c_int
a = my_dll.Infer_test(ctypes.c_void_p(aa))

#
# ci = my_dll.createInfer
# ci.restype = ctypes.c_void_p
# obj = ci()
# a1 = my_dll.Infer_test(ctypes.byref(obj))
# # 创建类的实例
# dll = ctypes.WinDLL(r"D:\04Bin\KDAI_Infer.dll")
# infer_instance = ctypes.POINTER(dll.CKDAI_Infer)()
# method = getattr(infer_instance.contents, "test")
# b = method()
path = b"D:/04Bin/model/KDAI_13_Side_BlackGray_1024_1024_1_EWMPS_0.1"

my_dll.Infer_LoadModel.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
p_str = ctypes.c_char_p(path.encode("utf-8"))
# result = my_dll.Infer_LoadModel(ctypes.byref(infer), 2, p_str, 1)
result = my_dll.Infer_LoadModel1(ctypes.c_void_p(aa), 2, p_str, 1)

resulta = my_dll.Infer_test1(ctypes.c_void_p(aa))
p_str = ctypes.c_char_p(b"D:/04Bin/img/img.jpg")
my_dll.Infer_PridictImg(ctypes.c_void_p(aa), p_str)
pass