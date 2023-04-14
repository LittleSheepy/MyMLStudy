import cv2
import time
# 加载图像
start_time = time.time()
image_path = r'D:\04DataSets\04\box.jpg'
img = cv2.imread(image_path)
# 创建ORB对象进行关键点检测和描述符生成
orb = cv2.ORB_create()
# 检测关键点和描绘符
keypoints, descriptors = orb.detectAndCompute(img, None)
# 在图像上绘制关键点
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None)
end_time = time.time()
total_time = end_time - start_time
print("运行时间为：", total_time, "秒")
# 显示结果图像
cv2.imshow('Image with Keypoints', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()