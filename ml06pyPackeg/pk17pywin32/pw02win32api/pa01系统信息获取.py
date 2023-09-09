import win32api
import win32con

print("")
print("开始")
win32api.Sleep(2000)      # 延迟5秒
print("2s后")
win32api.keybd_event(32, 0, 0, 0)  # 模拟按下空格键
win32api.keybd_event(32, 0, win32con.KEYEVENTF_KEYUP, 0)  # 模拟释放空格键
# win32api.MouseMove(100, 100)  # 移动到(100, 100)位置
# 设置鼠标位置
win32api.SetCursorPos((500, 300))
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)  # 模拟左键按下
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)    # 模拟左键释放