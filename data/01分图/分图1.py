from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import shutil


class ImageBrowser:
    def __init__(self, master):
        self.master = master
        self.image_canvas = Canvas(master, width=500, height=500)
        self.image_canvas.pack()
        self.current_image_path = None
        self.image_paths = []
        self.current_image_index = 0
        self.zoom_level = 1

        open_button = Button(master, text="Open", command=self.open_folder)
        open_button.pack()

        next_button = Button(master, text="Next", command=self.next_image)
        next_button.pack()

        prev_button = Button(master, text="Prev", command=self.prev_image)
        prev_button.pack()

        move_button = Button(master, text="Move", command=self.move_image)
        move_button.pack()

        self.image_canvas.bind("<MouseWheel>", self.zoom_image)

    def open_folder(self):
        folder_path = filedialog.askdirectory()
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".jpeg") or file_name.endswith(".png"):
                self.image_paths.append(os.path.join(folder_path, file_name))
        self.show_image(self.image_paths[0])

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.show_image(self.image_paths[self.current_image_index])

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.image_paths[self.current_image_index])

    def move_image(self):
        destination_folder = filedialog.askdirectory()
        if self.current_image_path:
            shutil.move(self.current_image_path, destination_folder)
            self.image_paths.remove(self.current_image_path)
            self.current_image_index -= 1
            self.next_image()

    def show_image(self, image_path):
        self.current_image_path = image_path
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)
        self.image_canvas.create_image(0, 0, anchor=NW, image=photo)
        self.image_canvas.image = photo

    def zoom_image(self, event):
        # Get the current image size
        image = Image.open(self.current_image_path)
        image_width, image_height = image.size

        # Get the mouse position relative to the image_canvas widget
        canvas_x = self.image_canvas.canvasx(event.x)
        canvas_y = self.image_canvas.canvasy(event.y)

        # Calculate the mouse position relative to the image
        image_x = canvas_x - (self.image_canvas.winfo_width() - image_width) / 2
        image_y = canvas_y - (self.image_canvas.winfo_height() - image_height) / 2

        # Calculate the relative position of the mouse in the image (between 0.0 and 1.0)
        relative_x = image_x / image_width
        relative_y = image_y / image_height

        # Zoom in or out
        if event.delta > 0:
            self.zoom_level += 0.5
        else:
            self.zoom_level -= 0.5

        # Resize the image
        image = image.resize((int(image_width * self.zoom_level), int(image_height * self.zoom_level)), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)

        # Calculate the new image position
        new_image_x = canvas_x - relative_x * photo.width()
        new_image_y = canvas_y - relative_y * photo.height()

        # Delete the old image and create a new one
        self.image_canvas.delete("all")
        self.image_canvas.create_image(new_image_x, new_image_y, image=photo, anchor="nw")
        self.image_canvas.image = photo


root = Tk()
my_gui = ImageBrowser(root)
root.mainloop()
