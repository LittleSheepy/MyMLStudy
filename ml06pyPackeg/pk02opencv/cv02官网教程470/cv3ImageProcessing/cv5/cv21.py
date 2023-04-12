# -*- coding: utf-8 -*-
"""
You will learn how to recover an out-of-focus image by Wiener filter
Author: Karpushin Vladislav, karpushin@ngs.ru, https://github.com/VladKarpushin
"""

import cv2
import numpy as np
import argparse

def help():
    print("2018-07-12")
    print("DeBlur_v8")
    print("You will learn how to recover an out-of-focus image by Wiener filter")

def calcPSF(outputImg, filterSize, R):
    h = np.zeros(filterSize, dtype=np.float32)
    point = (filterSize[1] // 2, filterSize[0] // 2)        # (240, 320)
    cv2.circle(h, point, R, 255, -1, 8)                     #
    summa = np.sum(h)                                       # 2246295.0
    outputImg[:] = h / summa

def fftshift(inputImg, outputImg):
    outputImg[:] = inputImg.copy()
    cx = outputImg.shape[1] // 2
    cy = outputImg.shape[0] // 2
    q0 = outputImg[0:cy, 0:cx]
    q1 = outputImg[0:cy, cx:2*cx]
    q2 = outputImg[cy:2*cy, 0:cx]
    q3 = outputImg[cy:2*cy, cx:2*cx]
    tmp = np.empty(q0.shape, dtype=q0.dtype)
    np.copyto(tmp, q0)
    np.copyto(q0, q3)
    np.copyto(q3, tmp)
    np.copyto(tmp, q1)
    np.copyto(q1, q2)
    np.copyto(q2, tmp)

def filter2DFreq(inputImg, outputImg, H):
    planes = [np.float32(inputImg.copy()), np.zeros(inputImg.shape, dtype=np.float32)]
    complexI = cv2.merge(planes)
    cv2.dft(complexI, complexI, cv2.DFT_SCALE)
    planesH = [np.float32(H.copy()), np.zeros(H.shape, dtype=np.float32)]
    complexH = cv2.merge(planesH)
    complexIH = cv2.mulSpectrums(complexI, complexH, 0) # , conjB=True
    cv2.idft(complexIH, complexIH)
    planes = cv2.split(complexIH)
    outputImg[:] = planes[0]

def calcWnrFilter(input_h_PSF, output_G, nsr):
    h_PSF_shifted = np.zeros_like(input_h_PSF)          # (480, 640)
    fftshift(input_h_PSF, h_PSF_shifted)
    planes = [np.float32(h_PSF_shifted.copy()), np.zeros(h_PSF_shifted.shape, dtype=np.float32)]
    complexI = cv2.merge(planes)
    cv2.dft(complexI, complexI)
    planes = cv2.split(complexI)
    denom = cv2.pow(cv2.magnitude(planes[0], planes[1]), 2)
    denom += nsr
    output_G[:] = planes[0] / denom

if __name__ == '__main__':
    dir_root = r"D:\02dataset\02opencv_data/"
    img_path = dir_root + "original.jpg"
    parser = argparse.ArgumentParser(description='Deblur an out-of-focus image using Wiener filter')
    parser.add_argument('--image', type=str, default=img_path, help='input image name')
    parser.add_argument('--R', type=int, default=53, help='radius')
    parser.add_argument('--SNR', type=int, default=5200, help='signal to noise ratio')
    args = parser.parse_args()

    help()
    imgIn = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)    # (480, 640)
    if imgIn is None:
        print("ERROR : Image cannot be loaded..!!")
        exit(-1)

    #imgOut = np.empty_like(imgIn)
    roi = (0, 0, imgIn.shape[0] & -2, imgIn.shape[1] & -2)      # (0, 0, 480, 640)

    imgOut = np.zeros(roi[2:], dtype=np.float32)        # (480, 640)
    #Hw calculation (start)
    Hw = np.zeros(roi[2:], dtype=np.float32)            # (480, 640)
    h = np.zeros(roi[2:], dtype=np.float32)             # (480, 640)
    calcPSF(h, roi[2:], args.R)                         # (480, 640)

    h_show = h / 0.00011352026 * 255
    cv2.imwrite("h_show.jpg", h_show)

    calcWnrFilter(h, Hw, 1.0 / float(args.SNR))

    max_ = Hw.max()
    Hw_show = Hw / max_ * 255
    cv2.imwrite("Hw_show.jpg", Hw_show)
    #Hw calculation (stop)

    # filtering (start)
    filter2DFreq(imgIn[roi[0]:roi[0]+roi[2], roi[1]:roi[1]+roi[3]], imgOut, Hw)
    # filtering (stop)

    imgOut = np.uint8(imgOut)
    cv2.normalize(imgOut, imgOut, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("result.jpg", imgOut)
