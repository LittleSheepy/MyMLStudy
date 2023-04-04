from tkinter import *
root = Tk()
def onClick():
    print("Button clicked")
var = StringVar(value="button1")
btn1 = Radiobutton(root, text="Button 1", variable=var, value="button1", command=onClick)
btn2 = Radiobutton(root, text="Button 2", variable=var, value="button2", command=onClick)
btn3 = Radiobutton(root, text="Button 3", variable=var, value="button3", command=onClick)
btn1.pack()
btn2.pack()
btn3.pack()
root.mainloop()
