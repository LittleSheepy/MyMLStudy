from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageDraw, ImageTk
import sys
import cv2
import numpy as np
# from PIL.Image import Image

from ..tk01myTools.tk01draw1 import ImageEditor

class ImageEditorLGSZ(ImageEditor):
    def __init__(self):
        super().__init__()
