from __future__ import print_function
import sys
import time

import numpy as np
import cv2
import cv2 as cv

""" kernel 滤波器核 """
# contrast enhancement 增加对比度
kernel_contrast = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], np.float32)  # kernel should be floating point type

# 边缘检测卷积核
kernel_edge_detection = np.array([[-1, -1, -1],
                                  [-1, 8, -1],
                                  [-1, -1, -1]], np.float32)

# Sharpen kernel 锐化卷积核
kernel_sharpen = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]], np.float32)

# Box blur kernel 盒式模糊卷积核
kernel_box_blur = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]], np.float32) / 9

# Gaussian blur kernel 高斯模糊卷积核
kernel_gaussian_blur = np.array([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]], np.float32) / 16

# Emboss kernel 浮雕卷积核
kernel_emboss = np.array([[-2, -1, 0],
                          [-1, 1, 1],
                          [0, 1, 2]], np.float32)

# Motion blur kernel 运动模糊卷积核
kernel_motion_blur = np.array([[1, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0],
                               [0, 0, 1, 0, 0],
                               [0, 0, 0, 1, 0],
                               [0, 0, 0, 0, 1]], np.float32) / 5

# Laplacian kernel 拉普拉斯卷积核
# 拉普拉斯卷积核通常用于图像处理中的边缘检测和图像锐化。它是一个二阶导数算子，可以突出图像中快速强度变化的区域。
kernel_laplacian = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], np.float32)

# Sobel X kernel Sobel X卷积核
kernel_sobel_x = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], np.float32)

# Sobel Y kernel Sobel Y卷积核
kernel_sobel_y = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], np.float32)

# Prewitt X kernel Prewitt X卷积核
kernel_prewitt_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]], np.float32)

# Prewitt Y kernel Prewitt Y卷积核
kernel_prewitt_y = np.array([[-1, -1, -1],
                             [0, 0, 0],
                             [1, 1, 1]], np.float32)

# Roberts X kernel Roberts X卷积核
kernel_roberts_x = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, -1]], np.float32)

# Roberts Y kernel Roberts Y卷积核
kernel_roberts_y = np.array([[0, 0, 0],
                             [0, 0, 1],
                             [0, -1, 0]], np.float32)

# Identity kernel 单位卷积核
kernel_identity = np.array([[0, 0, 0],
                             [0, 1, 0],
                             [0, 0, 0]], np.float32)
# Box blur kernel
kernel_box_blur = np.ones((3, 3), np.float32) / 9.0

# 矩形核，用于图像的腐蚀和膨胀操作。 彩色可用但无用
kernel_rect = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], np.uint8)
kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(3, 3))

#


def my_filter2D(img_src, kernel = kernel_contrast):
    ## [kern]
    kernel = kernel if kernel.all() else np.array([[ 0, -1,  0],
                                             [-1,  5, -1],
                                             [ 0, -1,  0]], np.float32)  # kernel should be floating point type
    ## [kern]
    ## [filter2D]
    img_dst = cv.filter2D(img_src, -1, kernel)
    return img_dst

## [basic_method]
def is_grayscale(my_image):
    return len(my_image.shape) < 3


def saturated(sum_value):
    if sum_value > 255:
        sum_value = 255
    if sum_value < 0:
        sum_value = 0

    return sum_value


def sharpen(my_image):
    if is_grayscale(my_image):
        height, width = my_image.shape
    else:
        my_image = cv.cvtColor(my_image, cv.CV_8U)
        height, width, n_channels = my_image.shape

    result = np.zeros(my_image.shape, my_image.dtype)
    ## [basic_method_loop]
    for j in range(1, height - 1):
        for i in range(1, width - 1):
            if is_grayscale(my_image):
                sum_value = 5 * my_image[j, i] - my_image[j + 1, i] - my_image[j - 1, i] \
                            - my_image[j, i + 1] - my_image[j, i - 1]
                result[j, i] = saturated(sum_value)
            else:
                for k in range(0, n_channels):
                    sum_value = 5 * my_image[j, i, k] - my_image[j + 1, i, k]  \
                                - my_image[j - 1, i, k] - my_image[j, i + 1, k]\
                                - my_image[j, i - 1, k]
                    result[j, i, k] = saturated(sum_value)
    ## [basic_method_loop]
    return result
## [basic_method]

def main(argv):
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + "lena.jpg"
    filename = img_path

    img_codec = cv.IMREAD_COLOR
    if argv:
        filename = sys.argv[1]
        if len(argv) >= 2 and sys.argv[2] == "G":
            img_codec = cv.IMREAD_GRAYSCALE

    src = cv.imread(cv.samples.findFile(filename), img_codec)

    if src is None:
        print("Can't open image [" + filename + "]")
        print("Usage:")
        print("mat_mask_operations.py [image_path -- default lena.jpg] [G -- grayscale]")
        return -1

    cv.namedWindow("Input", cv.WINDOW_AUTOSIZE)
    cv.namedWindow("Output", cv.WINDOW_AUTOSIZE)

    cv.imshow("Input", src)
    t = round(time.time())

    dst0 = sharpen(src)

    t = (time.time() - t)
    print("Hand written function time passed in seconds: %s" % t)

    cv.imshow("Output", dst0)
    cv.waitKey()

    t = time.time()
    ## [kern]
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)  # kernel should be floating point type
    ## [kern]
    ## [filter2D]
    dst1 = cv.filter2D(src, -1, kernel)
    # ddepth = -1, means destination image has depth same as input image
    ## [filter2D]

    t = (time.time() - t)
    print("Built-in filter2D time passed in seconds:     %s" % t)

    cv.imshow("Output", dst1)

    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0


if __name__ == "__main__":
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + "lena.jpg"
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #main(sys.argv[1:])
    img_bgr_filter = my_filter2D(img_bgr, kernel_laplacian)
    cv2.imshow("img_bgr_filter", cv2.hconcat([img_bgr, img_bgr_filter]))

    img_gray_filter = my_filter2D(img_gray, kernel_laplacian)
    img_gray_filter_sharpen = my_filter2D(img_gray, kernel_sharpen)
    cv2.imshow("img_gray_filter", cv2.hconcat([img_gray, img_gray_filter,img_gray_filter_sharpen]))
    cv2.waitKey(0)


