from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageDraw, ImageTk
import sys
import cv2
import numpy as np

sys.path.append("../../")
from ml00project.pj3distence.cv01distenceTest import distenceMeasure, imgdrawResult


def rgb2hex(rgb):
    hex_color = "#" + hex(rgb[0])[2:].zfill(2) + hex(rgb[1])[2:].zfill(2) + hex(rgb[2])[2:].zfill(2)
    return hex_color.upper()


class ImageEditor:
    image_draw: Image

    def __init__(self, width, height):
        # self.width = width
        # self.height = height
        self.file_path = r"D:\04DataSets\04/box.jpg"
        self.root = Tk()
        self.root.title("刘翠立的算法调试器")
        # self.root.geometry("800x800")
        self.brush_size = 15
        self.draw_color = (60, 60, 60)
        # 创建菜单栏
        menubar = Menu(self.root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="打开", command=self.open_image)
        filemenu.add_command(label="保存", command=self.save_image)
        menubar.add_cascade(label="文件", menu=filemenu)
        self.root.config(menu=menubar)

        # 创建工具栏
        toolbar = Frame(self.root)
        self.pen_btn = Button(toolbar, text="画笔", command=self.use_pen)
        self.pen_btn.pack(side=LEFT, padx=5, pady=5)

        self.color_btn = Button(toolbar, text="颜色", command=self.choose_color)
        self.color_btn.configure(bg=str(rgb2hex(self.draw_color)))
        self.color_btn.pack(side=LEFT, padx=5, pady=5)

        self.screen_btn = Button(toolbar, text="吸色", command=self.pick_color)
        self.screen_btn.pack(side=LEFT, padx=5, pady=5)

        self.reset_btn = Button(toolbar, text="重置", command=self.img_reset)
        self.reset_btn.pack(side=LEFT, padx=5, pady=5)

        toolbar.pack(side=TOP, fill=X)

        # 创建画布
        self.canvas_frame = Frame(self.root)
        self.canvas_frame.config(bg="red")
        # self.canvas1 = Canvas(self.canvas_frame, width=self.width, height=self.height)
        # self.canvas2 = Canvas(self.canvas_frame, width=self.width, height=self.height)
        # self.canvas3 = Canvas(self.canvas_frame, width=self.width, height=self.height)
        # self.canvas4 = Canvas(self.canvas_frame, width=self.width, height=self.height)
        self.canvas1 = Canvas(self.canvas_frame, bg='white')
        self.canvas2 = Canvas(self.canvas_frame, bg='white')
        self.canvas3 = Canvas(self.canvas_frame, bg='white')
        self.canvas4 = Canvas(self.canvas_frame, bg='white')
        self.canvas1.grid(row=0, column=0, padx=5, pady=5, sticky="nesw")
        self.canvas2.grid(row=0, column=1, padx=5, pady=5, sticky="nesw")
        self.canvas3.grid(row=1, column=0, padx=5, pady=5, sticky="nesw")
        self.canvas4.grid(row=1, column=1, padx=5, pady=5, sticky="nesw")
        self.canvas_frame.pack(expand=True, fill="both")
        self.canvas_frame.bind("<Configure>", self.resize_canvas)
        # self.root.bind("<Configure>", self.resize_frame)

        # 初始化画笔
        # self.image_show = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        # self.draw = ImageDraw.Draw(self.image_show)
        self.tool = None
        self.image_original = None
        self.image_draw = None
        self.image_show_ori = None
        self.photo = None

        # 绑定事件
        self.canvas1.bind("<B1-Motion>", self.paint)
        self.canvas2.bind("<B1-Motion>", self.paint)
        self.canvas3.bind("<B1-Motion>", self.paint)
        self.canvas4.bind("<B1-Motion>", self.paint)
        # self.canvas.bind("<ButtonRelease-1>", self.reset)

        # 图片信息
        self.window_w = 1000
        self.window_h = 1200
        self.img_w = 386
        self.img_h = 386
        self.canv_w = 386
        self.canv_h = 386
        self.ratio = 1

        self.init_img()

    def init_img(self):
        self.image_original = Image.open(self.file_path)
        self.img_w = self.image_original.size[0]
        self.img_h = self.image_original.size[1]
        self.resize_canvas()
        self.image_draw = self.image_original.copy()
        self.draw = ImageDraw.Draw(self.image_draw)
        self.recog_image()
        self.show_image()

    # 绘制，识别和现实
    def draw_image(self, x=None, y=None):
        if x and y:
            self.draw.ellipse((x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size),
                              fill=self.draw_color, outline=self.draw_color)
            self.recog_image()
        self.show_image()

    def recog_image(self):
        image_show_gray = cv2.cvtColor(np.array(self.image_draw), cv2.COLOR_RGB2GRAY)
        self.image_show_ori = cv2.cvtColor(np.array(self.image_draw), cv2.COLOR_RGB2BGR)  # PIL -> cv
        result_dict = distenceMeasure(image_show_gray)
        imgdrawResult(self.image_show_ori, result_dict)

    def show_image(self):
        if self.image_show_ori is None:
            return
        self.image_show = self.image_show_ori.copy()
        self.image_show = cv2.resize(self.image_show, (self.canv_w, self.canv_h))
        # self.image_show.resize((self.canv_w, self.canv_h), refcheck=False)
        self.image_show = Image.fromarray(cv2.cvtColor(self.image_show, cv2.COLOR_BGR2RGB))  # cv -> PIL
        self.photo = ImageTk.PhotoImage(self.image_show)
        self.canvas1.create_image(0, 0, image=self.photo, anchor=NW)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.image_original = Image.open(file_path)
            self.image_draw = self.image_original.copy()
            self.draw = ImageDraw.Draw(self.image_draw)
            self.image_show = self.image_original.copy()
            self.photo = ImageTk.PhotoImage(self.image_show)
            self.canvas1.create_image(0, 0, image=self.photo, anchor=NW)

    def save_image(self):
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",
                                                 filetypes=[("PNG 文件", "*.png"), ("JPG 文件", "*.jpg")])
        if file_path:
            self.image_draw.save(file_path)

    def use_pen(self):
        print(self.tool)
        self.tool = "pen"
        print("use_pen2", self.tool)

    def choose_color(self):
        color = colorchooser.askcolor(title="选择颜色")
        if color and color[0]:
            self.color_btn.configure(bg=str(color[1]))
            self.draw_color = color[0]

    def pick_color(self):
        print("pick_color1", self.tool)
        self.tool = "pick"
        print("pick_color2", self.tool)

    def paint(self, event):
        print("paint", self.tool)
        if self.tool == "pen":
            x, y = int(event.x / self.ratio), int(event.y / self.ratio)
            self.draw.ellipse((x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size),
                              fill=self.draw_color, outline=self.draw_color)
            # self.draw.line([event.x, event.y, event.x + 1, event.y + 1], fill=self.draw_color, width=20)

            image_show_gray = cv2.cvtColor(np.asarray(self.image_draw), cv2.COLOR_RGB2GRAY)
            image_show_bgr = cv2.cvtColor(np.asarray(self.image_draw), cv2.COLOR_RGB2BGR)  # PIL -> cv
            result_dict = distenceMeasure(image_show_gray)
            imgdrawResult(image_show_bgr, result_dict)
            self.image_show = Image.fromarray(cv2.cvtColor(image_show_bgr, cv2.COLOR_BGR2RGB))  # cv -> PIL

            # LineWhiteTop = resultDict["LineWhiteTop"]
            # k = LineWhiteTop[1] / LineWhiteTop[0]
            # b = LineWhiteTop[3] - k * LineWhiteTop[2]
            # draw = ImageDraw.Draw(self.image_show)
            # draw.line([0, int(b), 600, int(k * 600 + b)], fill=(255, 0, 0), width=1)
            self.photo = ImageTk.PhotoImage(self.image_show)
            self.canvas1.create_image(0, 0, image=self.photo, anchor=NW)
        elif self.tool == "pick":
            x, y = event.x, event.y
            print(self.image_show.getpixel((x, y)))
            self.color_btn.configure(bg=rgb2hex(self.image_show.getpixel((x, y))))
            self.draw_color = self.image_show.getpixel((x, y))

    def img_reset(self):
        self.init_img()
        pass

    def resize_canvas(self, event=None):
        if event:
            print("resize_canvas", event.width, event.height)
            self.window_w = event.width+4
            self.window_h = event.height+4

        w = self.window_w-10
        h = self.window_h-10
        self.ratio = min(w / (self.img_w * 2), h / (self.img_h * 2))
        self.canv_w = int(self.img_w * self.ratio) - 1
        self.canv_h = int(self.img_h * self.ratio) - 1
        canv_w = self.canv_w
        canv_h = self.canv_h
        self.canvas1.config(width=canv_w, height=canv_h - 10)
        self.canvas2.config(width=canv_w, height=canv_h - 10)
        self.canvas3.config(width=canv_w, height=canv_h - 10)
        self.canvas4.config(width=canv_w, height=canv_h - 10)
        self.show_image()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ImageEditor(800, 800)
    app.run()
