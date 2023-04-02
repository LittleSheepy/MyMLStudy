"""
    仿射变换
"""
import cv2 as cv
import numpy as np

# 仿射变换
def my_warpAffine():
    img_bgr = cv.imread(filename)
    # 设置3个点来计算仿射变换
    srcTri = np.array( [[0, 0], [100, 0], [100, 100]] ).astype(np.float32)
    dstTri = np.array( [[50, 50], [80, 60], [90, 90]] ).astype(np.float32)
    # 获取仿射变换矩阵
    warp_mat = cv.getAffineTransform(srcTri, dstTri)
    # 仿射变换
    img_affine = cv.warpAffine(img_bgr, warp_mat, (img_bgr.shape[1], img_bgr.shape[0]))
    return img_affine

# 旋转仿射变换
def myRotation_warpAffine():
    img_bgr = cv.imread(filename)
    h, w = img_bgr.shape[0], img_bgr.shape[1]
    # 获取仿射变换矩阵
    rot_mat = cv.getRotationMatrix2D(center=(w // 2, h // 2), angle=10, scale=0.6)
    img_rotation_affine = cv.warpAffine(img_bgr, rot_mat, (img_bgr.shape[1], img_bgr.shape[0]))
    return img_rotation_affine

if __name__ == '__main__':
    dir_root = r"D:\00myGitHub\opencv\samples\data/"
    filename = dir_root + 'lena.jpg'
    img_bgr = cv.imread(filename)
    img_affine = my_warpAffine()
    img_rotation_affine = myRotation_warpAffine()
    img_show = cv.hconcat([img_bgr, img_affine, img_rotation_affine])
    cv.imshow('img_show', img_show)
    cv.waitKey()
