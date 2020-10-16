import cv2 as cv
from pprint import pprint

flags = [i for i in dir(cv) if i.startswith('COLOR_')]
pprint(flags)