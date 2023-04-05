
import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk#图像控件

cap = cv2.VideoCapture(0)#创建摄像头对象
#界面画布更新图像
def tkImage():
    ref,frame=cap.read()
    #frame = cv2.flip(frame, 1) #摄像头翻转
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    pilImage=Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height),Image.ANTIALIAS)
    tkImage =  ImageTk.PhotoImage(image=pilImage)
    return tkImage
top = tk.Tk()
top.title('视频窗口')
top.geometry('900x600')
image_width = 600
image_height = 500
canvas = Canvas(top,bg = 'white',width = image_width,height = image_height )#绘制画布
Label(top,text = '这是一个视频！',font = ("黑体",14),width =15,height = 1).place(x =400,y = 20,anchor = 'nw')
canvas.place(x = 150,y = 50)
while True:
  pic = tkImage()
  canvas.create_image(0,0,anchor = 'nw',image = pic)
  top.update()
  top.after(1)

cap.release()
top.mainloop()
