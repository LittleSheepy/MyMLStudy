import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
import cv2
from PIL import Image, ImageDraw, ImageTk
# 识别函数
def recognition(img):
    print("识别图像")
    h,w = img.shape[0], img.shape[1]
    cv2.putText(img, "recognition", (50, 50), cv2.FONT_HERSHEY_COMPLEX,
                0.4, (255,0,0), 1)
    return img
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.width = 400
        self.height = 400
        self.master = master
        self.pack()
        self.create_widgets()
        self.video_capture = None
        self.recognitionFlg = True
        self.stop_video = False
        self.file_path = None
        self.master.title("异常行为检测")
    def create_widgets(self):
        toolbar = Frame(self)
        # 创建选择视频文件按钮
        self.choose_file = tk.Button(toolbar, text="选择视频文件", command=self.select_file)
        self.choose_file.pack(side="left")
        # 创建开始和停止识别按钮
        self.start_recognition = tk.Button(toolbar, text="开始识别", command=self.begin_recognition)
        self.start_recognition.pack(side="left")
        self.stop_recognition = tk.Button(toolbar, text="停止识别", command=self.end_recognition)
        self.stop_recognition.pack(side="left")
        # 创建打开和关闭摄像头按钮
        self.open_camera = tk.Button(toolbar, text="打开摄像头", command=self.activate_camera)
        self.open_camera.pack(side="left")
        self.close_camera = tk.Button(toolbar, text="关闭摄像头", command=self.deactivate_camera)
        self.close_camera.pack(side="left")
        toolbar.pack(side=TOP, fill=X)
        # 创建画布
        self.canvas = Canvas(self, width=self.width, height=self.height)
        self.canvas.pack(fill=BOTH, expand=YES)
        self.image_show = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        self.photo = ImageTk.PhotoImage(self.image_show)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def select_file(self):
        self.recognitionFlg = False
        self.file_path = filedialog.askopenfilename(title="选择视频")
        #file_path = filedialog.askopenfilename(title="选择视频", filetypes=[("视频文件", "*.MP4;*.jpeg;*.png;*.bmp")])
        if self.file_path:
            print("选择视频：",self.file_path)
            self.video_capture = cv2.VideoCapture(self.file_path)
            ret, frame = self.video_capture.read()
            self.show_canvas(frame)
    def show_canvas(self, img_cv):
        frame= Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.config(width=frame.size[0], height=frame.size[0])
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)

    def begin_recognition(self):
        self.recognitionFlg = True
        i = 0
        while self.recognitionFlg:
            if self.video_capture.isOpened():
                i = i+1
                ret, frame = self.video_capture.read()
                if ret:
                    frame = recognition(frame)
                    self.show_canvas(frame)
                    self.update()
                    self.after(1)
                    cv2.waitKey(40)
                else:
                    break
        # 实现开始识别功能
        pass
    def end_recognition(self):
        self.recognitionFlg = False
        # 实现停止识别功能
        pass
    def activate_camera(self):
        # 实现打开摄像头功能
        self.video_capture = cv2.VideoCapture(0)
        # ret, frame = self.video_capture.read()
        # self.show_canvas(frame)
        self.recognitionFlg = False
        self.stop_video = False
        while not self.stop_video:
            if self.video_capture.isOpened():
                ret, frame = self.video_capture.read()
                if ret:
                    if self.recognitionFlg:
                        frame = recognition(frame)
                    self.show_canvas(frame)
                    self.update()
                    self.after(1)
                    cv2.waitKey(40)
                else:
                    break
    def deactivate_camera(self):
        # 实现关闭摄像头功能
        self.stop_video = True
        self.image_show = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        self.photo = ImageTk.PhotoImage(self.image_show)
        self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.update()
root = tk.Tk()
app = Application(master=root)
app.mainloop()