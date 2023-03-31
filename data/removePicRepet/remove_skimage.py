#!/usr/bin/env python# -*- coding: utf-8 -*-
# @Time    : 2019/1/15 9:19
# @Author  : xiaodai
import os
import cv2
import os, shutil
from tqdm import tqdm
#from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# from skimage.measure import compare_ssim
# import shutil
# def yidong(filename1,filename2):
#     shutil.move(filename1,filename2)
def delete(filename1):
    os.remove(filename1)

def del1(path):
    img_path = path
    imgs_n = []
    num = []
    img_files = [os.path.join(rootdir, file) for rootdir, _, files in os.walk(path)
                 for file in files if
                 (file.endswith('.png'))]
    for currIndex, filename in enumerate(img_files):
        if not os.path.exists(img_files[currIndex]):
            print('not exist', img_files[currIndex])
            break
        img = cv2.imread(img_files[currIndex])
        img1 = cv2.imread(img_files[currIndex + 1])
        ssim = compare_ssim(img, img1, multichannel=True)
        if ssim > 0.9:
            imgs_n.append(img_files[currIndex + 1])
            print(img_files[currIndex], img_files[currIndex + 1], ssim)
        else:
            print('small_ssim', img_files[currIndex], img_files[currIndex + 1], ssim)
        currIndex += 1
        if currIndex >= len(img_files) - 1:
            break
    for image in imgs_n:
    # yidong(image, save_path_img)
        delete(image)


def del2(pathsrc, pathsave, pathrepet):
    for imgSrcName in os.listdir(pathsrc):
        imgSrc = cv2.imread(pathsrc + imgSrcName)
        moveFlg = True
        for imgSavename in tqdm(os.listdir(pathsave)):
            imgSave = cv2.imread(pathsave + imgSavename)
            ssim = compare_ssim(imgSrc, imgSave, multichannel=True)
            if ssim > 0.7:
                moveFlg = False
                continue
        if moveFlg:
            shutil.move(pathsrc + imgSrcName, pathsave + imgSrcName)
        else:
            shutil.move(pathsrc + imgSrcName, pathrepet + imgSrcName)



if __name__ == '__main__':
    pathsrc = r"D:\04DataSets\02\images_src/"
    pathsave = r"D:\04DataSets\02\images_save/"
    pathrepet = r"D:\04DataSets\02\images_repet/"
    del2(pathsrc, pathsave, pathrepet)
