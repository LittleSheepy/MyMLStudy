
import tkinter as tk
class MainApplication(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(expand=True, fill="both")
        # 创建菜单栏和文件菜单
        menu = tk.Menu(self.master)
        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label='New')
        file_menu.add_command(label='Open')
        file_menu.add_separator()
        file_menu.add_command(label='Exit', command=self.master.quit)
        menu.add_cascade(label='File', menu=file_menu)
        self.master.config(menu=menu)
        # 创建工具栏
        toolbar = tk.Frame(self.master, bd=1, relief=tk.RAISED)
        tk.Button(toolbar, text='Button 1').pack(side=tk.LEFT, padx=2, pady=2)
        tk.Button(toolbar, text='Button 2').pack(side=tk.LEFT, padx=2, pady=2)
        toolbar.pack(side=tk.TOP, fill=tk.X)
        # 创建一个Frame来放置四个画布
        self.canvas_frame = tk.Frame(self.master)
        self.canvas_frame.pack(expand=True, fill="both")
        # 创建四个画布并放置到Frame中，使用grid布局
        self.canvas1 = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas1.grid(row=0, column=0, sticky="nesw")
        self.canvas2 = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas2.grid(row=0, column=1, sticky="nesw")
        self.canvas3 = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas3.grid(row=1, column=0, sticky="nesw")
        self.canvas4 = tk.Canvas(self.canvas_frame, bg='white')
        self.canvas4.grid(row=1, column=1, sticky="nesw")
        # 配置网格布局
        # tk.Grid.columnconfigure(self.canvas_frame, 0, weight=5)
        # tk.Grid.columnconfigure(self.canvas_frame, 1, weight=5)
        # tk.Grid.rowconfigure(self.canvas_frame, 0, weight=5)
        # tk.Grid.rowconfigure(self.canvas_frame, 1, weight=5)

        self.canvas1.grid(row=0, column=0, padx=5, pady=5)
        self.canvas2.grid(row=0, column=1, padx=5, pady=5)
        self.canvas3.grid(row=1, column=0, padx=5, pady=5)
        self.canvas4.grid(row=1, column=1, padx=5, pady=5)
        # 配置画布大小随窗口大小缩放
        self.canvas_frame.bind("<Configure>", self.on_resize)
    def on_resize(self, event):
        print("on_resize", event.width, event.height)
        width = event.width/2-14
        height = event.height/2-14
        self.canvas1.config(width = width, height = height)
        self.canvas2.config(width = width, height = height)
        self.canvas3.config(width = width, height = height)
        self.canvas4.config(width = width, height = height)
root = tk.Tk()
app = MainApplication(master=root)
app.mainloop()