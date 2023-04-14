from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageDraw, ImageTk
import sys
import cv2
import numpy as np

sys.path.append("../../")
from ml00project.pj3distence.cv01distenceTest import distenceMeasure, imgdrawResult

sys.path.append("../../")
from ml00project.pj2LG.LGPostProcess import CPostProcessor, mask2defectList # , imgdrawResult


def rgb2hex(rgb):
    hex_color = "#" + hex(rgb[0])[2:].zfill(2) + hex(rgb[1])[2:].zfill(2) + hex(rgb[2])[2:].zfill(2)
    return hex_color.upper()


class ImageEditor:
    image_draw_mask: Image

    def __init__(self, width, height):
        # self.width = width
        # self.height = height
        dir_root = r"D:\04DataSets\ningjingLG\all1\/"
        img_first_name = "black_0074690_CM1_"
        self.file_path = r"D:\04DataSets\ningjingLG\all\black_0074690_CM1_1.bmp"
        self.root = Tk()
        self.root.title("刘翠立的算法调试器")
        self.root.geometry("1000x1200")
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

        self.reset_btn = Button(toolbar, text="识别", command=self.recog_image)
        self.reset_btn.pack(side=LEFT, padx=5, pady=5)

        self.label = Label(toolbar, text="正确")
        self.label.pack(side=LEFT, padx=5, pady=5)

        toolbar.pack(side=TOP, fill=X)


        # 创建画布
        self.canvas_frame = Frame(self.root)
        self.canvas_frame.config(bg="red")
        self.canvas = [Canvas]*4
        for i in range(4):
            self.canvas[i] = Canvas(self.canvas_frame, bg='white', name=str(i))
            self.canvas[i].bind("<B1-Motion>", self.paint)
            row = i//2
            column = i%2
            self.canvas[i].grid(row=row, column=column, sticky="nesw")
        self.canvas_frame.pack(expand=True, fill="both")
        self.canvas_frame.bind("<Configure>", self.resize_canvas)
        # self.root.bind("<Configure>", self.resize_frame)

        # 初始化画笔
        # self.image_show = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        # self.draw = ImageDraw.Draw(self.image_show)
        self.tool = "pen"
        self.image_original = [None]*4
        self.image_original_cv = [None]*4
        self.image_draw_mask = [None] * 4
        self.draw = [ImageDraw.ImageDraw]*4
        self.draw_rgb = [ImageDraw.ImageDraw]*4
        self.image_show_ori = [None]*4
        self.image_show = [None]*4
        self.photo = [None]*4

        # 图片信息
        self.window_w = 1000
        self.window_h = 1200
        self.img_w = 386
        self.img_h = 386
        self.canv_w = 386
        self.canv_h = 386
        self.ratio = 1

        self.init_img()
        self.defectListList = [[]]*4
        self.pp = CPostProcessor()

    def init_img(self):
        path_split = self.file_path.split(".")
        for i in range(4):
            file_path = path_split[0][:-1] + str(i + 1) + "." + path_split[1]

            self.image_original[i] = Image.open(file_path)
            self.image_original_cv[i] = cv2.cvtColor(np.array(self.image_original[i]), cv2.COLOR_RGB2BGR)
            self.img_w = self.image_original[i].size[0]
            self.img_h = self.image_original[i].size[1]
            self.image_draw_mask[i] = Image.fromarray(np.zeros(self.image_original[i].size, dtype=np.uint8))
            self.draw[i] = ImageDraw.Draw(self.image_draw_mask[i])
            self.image_show_ori[i] = self.image_original[i].copy()
            self.draw_rgb[i] = ImageDraw.Draw(self.image_show_ori[i])
            # self.image_show[i] = self.image_draw_mask[i].copy()
            # self.photo[i] = ImageTk.PhotoImage(self.image_show[i])
            # self.canvas[i].create_image(0, 0, image=self.photo[i], anchor=NW)

    # 绘制，识别和现实
    def draw_image(self, i, x=None, y=None):
        if x and y:
            self.draw[i].ellipse((x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size),
                              fill=255, outline=255)
            self.draw_rgb[i].ellipse((x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size),
                              fill=(255,)*3, outline=(255,)*3)
            self.recog_image(i)
        self.show_image(i)

    def recog_image(self, i=None):
        if i:
            self.defectListList[i] = mask2defectList(np.array(self.image_draw_mask[i]))
        result = self.pp.Process(self.image_original_cv, self.defectListList)
        self.label.config(text=str(result))
        print(str(result))
        #imgdrawResult(self.image_show_ori[i], result_dict)

    def show_image(self, i):
        if self.image_show_ori[i] is None:
            return
        self.image_show[i] = self.image_show_ori[i].copy()
        self.image_show[i] = self.image_show[i].resize((self.canv_w, self.canv_h))
        # self.image_show.resize((self.canv_w, self.canv_h), refcheck=False)
        # self.image_show[i] = Image.fromarray(cv2.cvtColor(self.image_show[i], cv2.COLOR_BGR2RGB))  # cv -> PIL
        self.photo[i] = ImageTk.PhotoImage(self.image_show[i])
        self.canvas[i].create_image(0, 0, image=self.photo[i], anchor=NW)

    def open_image(self):
        file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("图片文件", "*.jpg;*.jpeg;*.png;*.bmp")])
        if file_path:
            self.file_path = file_path
            self.init_img()

    def save_image(self):
        file_path = filedialog.asksaveasfilename(title="保存图片", defaultextension=".png",
                                                 filetypes=[("PNG 文件", "*.png"), ("JPG 文件", "*.jpg")])
        if file_path:
            self.image_draw_mask.save(file_path)

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
            i = int(event.widget._name)
            x, y = int(event.x / self.ratio), int(event.y / self.ratio)
            self.draw_image(i, x, y)
        elif self.tool == "pick":
            x, y = event.x, event.y
            self.color_btn.configure(bg=rgb2hex(self.image_show.getpixel((x, y))))
            self.draw_color = self.image_show.getpixel((x, y))

    def img_reset(self):
        self.init_img()
        for i in range(4):
            self.show_image(i)

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
        for i in range(4):
            self.canvas[i].config(width=canv_w, height=canv_h)
            self.show_image(i)

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    app = ImageEditor(800, 800)
    app.run()
