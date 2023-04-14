from pprint import pprint
import cv2 as cv
import numpy as np


mag, ang = cv.cartToPolar((1,0), (1,0))
pprint([mag, ang])
print("*"*20)
mag, ang = cv.cartToPolar((1, 0, -1, 0), (0, 1, 0, -1))
pprint([mag, ang])

