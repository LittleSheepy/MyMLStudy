import win32gui, win32con
import time

hwnd = win32gui.FindWindow(None, "程序和功能")    # 可以根据窗口类名和窗口标题查找窗口句柄
rect = win32gui.GetWindowRect(hwnd)             # 获取窗口的位置和大小信息

win32gui.SetForegroundWindow(hwnd)              # 设为前台窗口
time.sleep(5)
win32gui.ShowWindow(hwnd,win32con.SW_MAXIMIZE)      # 最大化窗口
time.sleep(2)
# 设为后台窗口
win32gui.SetWindowPos(hwnd, win32con.HWND_BOTTOM, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_NOACTIVATE)

print("\n", rect)