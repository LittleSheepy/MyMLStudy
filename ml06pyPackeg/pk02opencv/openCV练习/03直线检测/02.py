import cv2
import numpy as np

def line_detect(image):
  # 将图片转换为HSV
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  # 设置阈值
  lowera = np.array([0, 0, 221])
  uppera = np.array([180, 30, 255])
  mask1 = cv2.inRange(hsv, lowera, uppera)
  kernel = np.ones((3, 3), np.uint8)

  # 对得到的图像进行形态学操作（闭运算和开运算）
  mask = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, kernel) #闭运算：表示先进行膨胀操作，再进行腐蚀操作
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  #开运算：表示的是先进行腐蚀，再进行膨胀操作

  # 绘制轮廓
  edges = cv2.Canny(mask, 50, 150, apertureSize=3)
  # 显示图片
  cv2.imshow("edges", edges)
  # 检测白线  这里是设置检测直线的条件，可以去读一读HoughLinesP()函数，然后根据自己的要求设置检测条件
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 80,minLineLength=10,maxLineGap=10)
  print ("lines=",lines)
  print ("========================================================")
  i=1
  # 对通过霍夫变换得到的数据进行遍历
  for line in lines:
    # newlines1 = lines[:, 0, :]
    print ("line["+str(i-1)+"]=",line)
    x1,y1,x2,y2 = line[0]  #两点确定一条直线，这里就是通过遍历得到的两个点的数据 （x1,y1）(x2,y2)
    cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)   #在原图上画线
    # 转换为浮点数，计算斜率
    x1 = float(x1)
    x2 = float(x2)
    y1 = float(y1)
    y2 = float(y2)
    print ("x1=%s,x2=%s,y1=%s,y2=%s" % (x1, x2, y1, y2))
    if x2 - x1 == 0:
        print ("直线是竖直的")
        result=90
    elif (y2 - y1) == 0 :
      print ("直线是水平的")
      result=0
    else:
      # 计算斜率
      k = -(y2 - y1) / (x2 - x1)
      # 求反正切，再将得到的弧度转换为度
      result = np.arctan(k) * 57.29577
      print ("直线倾斜角度为：" + str(result) + "度")
    i = i+1
  #   显示最后的成果图
  cv2.imshow("line_detect",image)
  return result

if __name__ == '__main__':
  # 读入图片
  src = cv2.imread(r'D:\04DataSets\ningjingLG\black\black_0074690_CM1_1.bmp')
  # 设置窗口大小
  cv2.namedWindow("input image", cv2.WINDOW_AUTOSIZE)
  # 显示原始图片
  cv2.imshow("input image", src)
  # 调用函数
  line_detect(src)
  cv2.waitKey(0)