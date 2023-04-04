
import cv2
import numpy as np
import matplotlib.pyplot as plt
# 计算掩码的直方图
def calchist_for_mask(img):

    mask = np.zeros(img.shape, np.uint8)
    mask[200:400, 200:400] = 255

    histMI = cv2.calcHist([img], [0], mask, [256], [0, 255])
    histImage = cv2.calcHist([img], [0], None, [256], [0, 255])

    plt.plot(histMI, color="r")
    #plt.savefig("result_mask.jpg")
    plt.show()
def erode_demo(pic):
    #gray = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY) # 原图片类型转换为灰度图像
    #ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dst = cv2.erode(pic, kernel)
    return dst

def dilate_demo(binary):
    #gray = cv2.cvtColor(pic,cv2.COLOR_RGB2GRAY)
    #ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV)
    #cv2.imshow("binary", binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dst = cv2.dilate(binary, kernel)
    return dst

# 加载图像
image_path = r'D:\04DataSets\04\box_center2.jpg'
img = cv2.imread(image_path, 0)
# 对图像应用Sobel算子
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = np.maximum(laplacian, 0)
laplacian = laplacian / abs(laplacian.max()) * 255
min_ = laplacian.min()
max_ = laplacian.max()
mean_ = laplacian.mean()
laplacian = np.where(laplacian > (max_+mean_)/2, laplacian, 0)
hist1, bins = np.histogram(laplacian, int(max_), [min_, max_])
plt.plot(hist1[1:])
plt.xlim([0, max_])
plt.show()
#calchist_for_mask(laplacian)
#laplacian = laplacian + abs(laplacian.min())

# 计算梯度值
grad = cv2.magnitude(sobelx, sobely)
min2_ = grad.min()
max2_ = grad.max()
mean2_ = grad.mean()
grad = np.where(grad > (min2_+mean2_)/2, laplacian, 0)
hist2, bins = np.histogram(grad, int(max_), [mean2_, max2_])
grad = (grad/max2_)*255
plt.plot(hist2[1:])
plt.xlim([0, max_])
plt.show()
# 显示图像和梯度图
cv2.imshow('image', img)
cv2.imshow('gradient', laplacian)
laplacian = dilate_demo(laplacian)
laplacian = dilate_demo(laplacian)
laplacian = erode_demo(laplacian)
laplacian = erode_demo(laplacian)
laplacian = erode_demo(laplacian)
laplacian = dilate_demo(laplacian)
# laplacian = dilate_demo(laplacian)
# laplacian = erode_demo(laplacian)
# laplacian = erode_demo(laplacian)
# laplacian = erode_demo(laplacian)
# laplacian = erode_demo(laplacian)
# laplacian = dilate_demo(laplacian)
# laplacian = dilate_demo(laplacian)
# laplacian = erode_demo(laplacian)
# laplacian = erode_demo(laplacian)
cv2.imshow('gradient2laplacian', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()