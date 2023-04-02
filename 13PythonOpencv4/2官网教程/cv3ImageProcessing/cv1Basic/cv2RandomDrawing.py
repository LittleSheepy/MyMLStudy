"""
    随机画图
"""
import cv2
import cv2 as cv
import numpy as np
import random

#from PIL import Image,ImageFont,ImageDraw

# 随机生成线段
def Drawing_Random_Lines(image):
    h, w = image.shape[0], image.shape[1]
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0,h) )
    (x2, y2) = ( random.randint(0, w), random.randint(0, h))
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    cv2.line(image,(x1,y1 ), (x2, y2),RGB, random.randint(1, 10), 8)

def Drawing_Random_circle(image):
    h, w = image.shape[0], image.shape[1]
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0,h) )
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    radius = random.randint(5, 15)
    cv2.circle(image,(x1,y1),radius,RGB,-1)
    return image

def Drawing_Random_rect(image):
    h, w = image.shape[0], image.shape[1]
    w1 = int(w/5)
    h1 = int(h/5)
    (x1,y1 ) = ( random.randint(0, w) ,random.randint(0, h) )
    (x2, y2) = ( random.randint(0, w1), random.randint(0, h1))
    RGB=(random.randint(0, 255), random.randint(0, 255),random.randint(0, 255))
    cv2.rectangle(image,(x1,y1),(x2, y2),RGB,1)
    return image
if __name__ == "__main__":
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    for i in range(3):
        Drawing_Random_rect(image)

    cv.imshow("Random_Lines", image)
    cv.waitKey(0)