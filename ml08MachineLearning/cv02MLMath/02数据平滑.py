import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


# 移动平均：求窗口内的平均值
def MovingAverage():
    # 定义移动平均窗口大小
    window_size = 3

    # 计算简单移动平均
    sma = np.convolve(data, np.ones(window_size) / window_size, mode='valid')

    # 绘制移动平均曲线
    plt.plot(np.arange(window_size - 1, len(data)), sma, label="移动平均", color='red')


# EMA:
def ExponentialMovingAverage():
    # 定义平滑参数（通常称为平滑因子）
    alpha = 0.1

    # 计算EMA
    ema = [data[0]]  # 初始EMA值等于第一个数据点
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])

    # 绘制原始数据和EMA曲线
    plt.plot(ema, label="EMA", color='green')

def LowpassFiltering():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import butter, lfilter

    # 生成示例数据
    fs = 1000  # 采样频率
    t = np.linspace(0, 5, 5 * fs, endpoint=False)
    data = 5 * np.sin(2 * np.pi * 3 * t) + 2 * np.sin(2 * np.pi * 50 * t)

    # 设计巴特沃斯低通滤波器
    cutoff_freq = 10  # 截止频率（以Hz为单位）
    nyquist_freq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(4, normal_cutoff, btype='low', analog=False)

    # 使用滤波器平滑数据
    smoothed_data = lfilter(b, a, data)

    # 绘制原始数据和平滑后的数据
    plt.figure(figsize=(10, 6))
    plt.plot(t, data, label="原始数据", color='blue')
    plt.plot(t, smoothed_data, label="低通滤波后的数据", color='red')
    plt.legend()
    plt.title("巴特沃斯低通滤波器示例")
    plt.xlabel("时间 (秒)")
    plt.ylabel("数值")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # 生成示例数据
    # data = np.array([10, 15, 12, 18, 20, 14, 16, 22, 19, 25])
    # data = np.array([10, 15, 12, 18, 11, 14, 10, 17, 11, 16])
    data = np.array([0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5, 0, 10, 20, 10, 0, 10, 20, 30, 5])
    # plt.figure(figsize=(10, 6))
    # plt.plot(data, label="原始数据", marker='o', color='blue')
    # MovingAverage()                     # 移动平均
    # ExponentialMovingAverage()          # EMA
    # plt.legend()
    # plt.title("简单移动平均示例")
    # plt.xlabel("数据点")
    # plt.ylabel("数值")
    # plt.grid(True)
    # plt.show()
    LowpassFiltering()