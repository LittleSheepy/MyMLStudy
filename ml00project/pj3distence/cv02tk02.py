from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk, ImageDraw


class PaintCanvas:
    def __init__(self, root, image_path):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.draw = ImageDraw.Draw(self.image)
        self.canvas = Canvas(root, width=self.image.width, height=self.image.height)
        self.canvas.pack()
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=NW, image=self.tk_image)
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_rectangle(x, y, x + 15, y + 15, fill="black")
        self.draw.rectangle((x, y, x + 15, y + 15), fill="black")

    def save_image(self):
        filename = asksaveasfilename(defaultextension='.png', filetypes=[("PNG files", ".png"), ("All files", "*.*")])
        if filename:
            self.image.save(filename)


class PaintApplication:
    def __init__(self, root):
        self.root = root
        self.menu = Menu(root)
        self.root.configure(menu=self.menu)
        self.file_menu = Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Open", command=self.open_image)
        self.file_menu.add_command(label="Save", command=self.save_image)
        self.paint_canvas = None

    def open_image(self):
        filename = askopenfilename(filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if filename:
            if self.paint_canvas:
                self.paint_canvas.canvas.pack_forget()
            self.paint_canvas = PaintCanvas(self.root, filename)

    def save_image(self):
        if self.paint_canvas:
            self.paint_canvas.save_image()


if __name__ == "__main__":
    root = Tk()
    root.title("Paint Application")
    app = PaintApplication(root)
    root.mainloop()