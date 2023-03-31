import math
import cv2
import numpy as np

point_list = []
img_show_windows = 'show'
image_path = r'D:\04DataSets\04\box.jpg'

def myMouseCallBbackFunc(event, x, y, flags, params_img):
    if event == cv2.EVENT_LBUTTONDOWN:  # 鼠标点击事件
        # 在鼠标的点击点画一个实心圆, 表示在这里取这个坐标点
        cv2.circle(img=params_img, center=(x, y), radius=5, color=(255, 0, 0), thickness=cv2.FILLED)
        # 将该点的位置信息保留记录
        point_list.append([x, y])
        print(point_list)

        # 根据实际情况画出角度的直线
        if len(point_list) == 3:
            p1, p2, p3 = point_list
            cv2.line(params_img, p1, p2, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
            cv2.line(params_img, p1, p3, color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)


def getEdgeTheta(pt_from, pt_to):
    k = abs(pt_to[1] - pt_from[1]) / (pt_to[0] - pt_from[0])
    theta = abs(math.atan(k) * 180 / math.pi)

    if pt_to[1] <= pt_from[1]:  # to点在from点的上面
        if pt_to[0] > pt_from[0]:  # 第一象限
            theta = theta
        elif pt_to[0] == pt_from[0]:  # 垂直
            theta = 90
        else:  # 第二象限
            theta = 180 - theta

    else:  # to点在from点的下面
        if pt_to[0] < pt_from[0]:  # 第三象限
            theta = theta + 180
        elif pt_to[0] == pt_from[0]:  # 垂直
            theta = 270
        else:  # 第二象限
            theta = 360 - theta

    print('theta:{}'.format(theta))
    return theta


def getOneAngle(img):
    # first point is center
    p1, p2, p3 = point_list
    len_a = math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))
    len_b = math.sqrt(math.pow((p3[0] - p1[0]), 2) + math.pow((p3[1] - p1[1]), 2))
    len_c = math.sqrt(math.pow((p3[0] - p2[0]), 2) + math.pow((p3[1] - p2[1]), 2))

    print('len_a:{}, len_b:{}, len_c:{}'.format(len_a, len_b, len_c))

    # 求出c对应的角的角度，是使用余弦定理做的
    angle = math.acos((math.pow(len_a, 2) + math.pow(len_b, 2) - math.pow(len_c, 2)) / (2 * len_a * len_b))
    angle = angle * 180 / math.pi
    print('get angle: {}'.format(angle))

    # 在图上画出表示角度的圆弧出来
    theta_a = getEdgeTheta(p1, p2)
    theta_b = getEdgeTheta(p1, p3)

    # 计算起始角度和结束角度，角度值维持在180度以内
    maxval = max(int(theta_a), int(theta_b))
    minval = min(int(theta_a), int(theta_b))
    if minval + 180 < maxval:
        startAngle, endAngle = maxval, minval
    else:
        startAngle, endAngle = minval, maxval
    print('startAngle:{}, endAngle:{}'.format(startAngle, endAngle))

    radius = int(min(len_a, len_b) / 2)  # 计算画弧线的半径

    # 画出弧线
    cv2.ellipse(img=img, center=p1, axes=(radius, radius),
                angle=int(360 - int(endAngle)), startAngle=0, endAngle=int(angle),
                color=(0, 255, 0), thickness=1)

    cv2.putText(img=img, text='{}'.format("%.2f" % angle), org=(p1[0]-10, p1[1]), fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=3)  # 在图上画出这个


if __name__ == '__main__':
    img = cv2.imread(image_path)

    #cv2.namedWindow(img_show_windows)  # 定义一个window
    #cv2.setMouseCallback(img_show_windows, myMouseCallBbackFunc, img)  # 设置这个窗口的鼠标事件回调函数

    while True:
        cv2.imshow(img_show_windows, img)  # 在这个window上画图
        # 其实点击事件还在，设置这个窗口的鼠标事件回调函数的目的，这里主要是更新这个新读取的图像这个参数
        cv2.setMouseCallback(img_show_windows, myMouseCallBbackFunc, img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q键清空所有的保留的点，重新展示一个新的图像
            point_list.clear()  # 清空所有的点
            img = cv2.imread(image_path)  # 重新获取图像

        if len(point_list) == 3:  # 三个点够了，我们可以得到三个点的角度。
            getOneAngle(img)  # 得到角度，这里是使用余弦定理来做的。
            point_list.clear()  # 清空所有的点



