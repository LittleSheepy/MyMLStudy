import cv2

# 创建空白画布
canvas = None

drawing = False

# 定义滑动条回调函数
def onChange(value):
    # 将滑动条的值转换为浮点数作为倍数
    alpha = float(value) / 100

    # 使用相关系数计算正反比例
    beta = 1.0 - alpha

    # 将两个图像相加，使用alpha和beta参数加权
    dst = cv2.addWeighted(img1, alpha, img2, beta, 0.0)

    # 显示加权后的图像
    cv2.imshow('Blended Image', dst)

img_path = r"D:\04DataSets\04\box.jpg"
# 加载两张图片
img1 = cv2.imread(img_path)
img2 = cv2.imread(img_path)

# 调整两张图片的大小为相同尺寸
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# 创建一个名为Blended Image的窗口，用来显示加权后的图像
cv2.namedWindow('Blended Image')

# 创建一个名为Image 1的窗口，用来显示第一张图片
cv2.namedWindow('Image 1')

# 创建一个名为Image 2的窗口，用来显示第二张图片
cv2.namedWindow('Image 2')

# 显示第一张图片
cv2.imshow('Image 1', img1)

# 显示第二张图片
cv2.imshow('Image 2', img2)

# 创建一个滑动条，用来调整合成比例
cv2.createTrackbar('Alpha', 'Blended Image', 0, 100, onChange)

# 进入消息循环，等待用户输入
while True:

    # 获取键盘输入
    key = cv2.waitKey(1000)
    print(key, ord('z'))
    # 如果是空格键，创建一张新的空白画布
    if key == ord(' '):
        canvas = cv2.imread('image1.jpg')
        canvas = cv2.resize(canvas, (500, 500))

    # 如果是c键，清空画布
    elif key == ord('c'):
        canvas = None

    # 如果是s键，保存画布
    elif key == ord('s'):
        if canvas is not None:
            cv2.imwrite('output.jpg', canvas)

    # 如果是鼠标左键，开始绘制
    elif key == ord('z'):
        if canvas is not None:
            # 定义绘制状态

            # 定义绘制颜色和大小
            color = (255, 255, 255)
            size = 10


            # 创建一个鼠标回调函数，实现绘制功能
            def onMouse(event, x, y, flags, param):
                global drawing
                # 如果是鼠标左键按下，开始绘制
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True

                # 如果鼠标移动，且正在绘制，绘制一条线段
                elif event == cv2.EVENT_MOUSEMOVE and drawing == True:
                    cv2.line(canvas, (x - size, y - size), (x + size, y + size), color, 2 * size)
                    cv2.line(canvas, (x - size, y + size), (x + size, y - size), color, 2 * size)

                # 如果鼠标左键松开，结束绘制
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False


            # 安装鼠标回调函数
            cv2.setMouseCallback('Blended Image', onMouse)

    # 如果用户按下ESC键，退出程序
    elif key == 27:
        break

# 关闭所有窗口
cv2.destroyAllWindows()