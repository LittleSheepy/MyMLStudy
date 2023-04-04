
from tkinter import *
from tkinter import filedialog
from tkinter import colorchooser
from PIL import Image, ImageDraw, ImageTk

from cv01distenceTest import distenceMeasure

class ImageEditor:
    def __init__(self, master):
        self.master = master
        self.canvas = Canvas(self.master, width=500, height=500, bg='white')
        self.canvas.pack(side=LEFT, padx=10, pady=10)
        self.image = None
        self.draw = None
        self.color = 'black'
        self.brush_size = 5
        menu = Menu(self.master)
        self.master.config(menu=menu)
        file_menu = Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        color_menu = Menu(menu)
        menu.add_cascade(label="Color", menu=color_menu)
        color_menu.add_command(label="Pick Color", command=self.pick_color)
        brush_menu = Menu(menu)
        menu.add_cascade(label="Brush", menu=brush_menu)
        brush_menu.add_command(label="Size 5", command=lambda: self.set_brush_size(5))
        brush_menu.add_command(label="Size 10", command=lambda: self.set_brush_size(10))
        brush_menu.add_command(label="Size 15", command=lambda: self.set_brush_size(15))
    def open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image File", ".jpg .png .gif")])
        if path:
            self.image = Image.open(path)
            self.draw = ImageDraw.Draw(self.image)
            self.display_image()
    def save_image(self):
        if self.image:
            path = filedialog.asksaveasfilename(filetypes=[("Image File", ".jpg .png .gif")])
            if path:
                self.image.save(path)
    def pick_color(self):
        color = colorchooser.askcolor()[1]
        if color:
            self.color = color
    def set_brush_size(self, size):
        self.brush_size = size
    def display_image(self):
        self.canvas.delete(ALL)
        image_tk = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=image_tk, anchor=NW)
        self.canvas.image_tk = image_tk # keep a reference
    def on_canvas_click(self, event):
        if self.draw:
            x, y = event.x, event.y
            self.draw.ellipse((x - self.brush_size, y - self.brush_size, x + self.brush_size, y + self.brush_size), fill=self.color, outline=self.color)
            self.display_image()
root = Tk()
app = ImageEditor(root)
app.canvas.bind("<B1-Motion>", app.on_canvas_click)
root.mainloop()