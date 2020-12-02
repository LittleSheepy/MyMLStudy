# coding:utf-8
import os

from PIL import Image


# bmp 转换为jpg
def bmpToJpg(file_path):
 for fileName in os.listdir(file_path):
  # print(fileName)
  newFileName = fileName[0:fileName.find(".")]+".jpg"
  print(newFileName)
  im = Image.open(file_path+"\\"+fileName)
  im.save(file_path+"\\"+newFileName)


# 删除原来的位图
def deleteImages(file_path, imageFormat):
 command = "del "+file_path+"\\*."+imageFormat
 os.system(command)


def main():
 file_path = "D:/01sheepy\work/baojie_ocr\pendianword\img/"
 bmpToJpg(file_path)
 deleteImages(file_path, "bmp")


if __name__ == '__main__':
 main()
