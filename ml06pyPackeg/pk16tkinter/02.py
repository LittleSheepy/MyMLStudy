

# Import necessary modules
import tkinter as tk
import cv2
from PIL import Image, ImageTk

# Create a Tkinter window
window = tk.Tk()

# Create a canvas to display the video
canvas = tk.Canvas(window)
canvas.pack()

# Load the video using OpenCV
video = cv2.VideoCapture(r'D:\01/not_fall.mp4')

# Define a function to update the video frames
def update_video():
    # Read a frame from the video
    ret, frame = video.read()
    if ret:
        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert the frame to a Tkinter-compatible image
        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image)
        # Update the canvas with the new image
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        # Schedule the function to be called again in 15 milliseconds
        window.after(15, update_video)

# Call the function to start updating the video frames
update_video()

# Start the Tkinter event loop
window.mainloop()