
import tkinter as tk
import cv2
from PIL import Image, ImageTk
class GUI:
    def __init__(self, master):
        self.master = master
        self.canvas1 = tk.Canvas(self.master, width=400, height=300)
        self.canvas2 = tk.Canvas(self.master, width=400, height=300)
        self.canvas1.grid(row=0, column=0)
        self.canvas2.grid(row=0, column=1)
        self.cap = cv2.VideoCapture(0)
        self.update()
    def update(self):
        ret, frame = self.cap.read()
        if ret:
            # 显示第一幅图像
            image1 = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            photo1 = ImageTk.PhotoImage(image1)
            self.canvas1.create_image(0, 0, image=photo1, anchor=tk.NW)
            self.canvas1.image = photo1
            # 显示第二幅图像
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image2 = Image.fromarray(gray)
            photo2 = ImageTk.PhotoImage(image2)
            self.canvas2.create_image(0, 0, image=photo2, anchor=tk.NW)
            self.canvas2.image = photo2
        self.master.after(20, self.update)
if __name__ == '__main__':
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()