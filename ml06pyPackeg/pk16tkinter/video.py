# Import necessary libraries
import cv2

# Read video file
cap = cv2.VideoCapture(r'D:\00myGitHub\00MyMLStudy\ml01tkinter\fall_detection_actions-master\Data\/not_fall.mp4')

# Loop through each frame of the video
while cap.isOpened():
    # Read current frame
    ret, frame = cap.read()

    # Check if frame was successfully read
    if not ret:
        break

    # Display the frame
    cv2.imshow('frame', frame)

    # Wait for 25 milliseconds before displaying the next frame
    # This is equivalent to a frame rate of 40 frames per second
    # The waitKey function returns the ASCII value of the key pressed
    # If the 'q' key is pressed, exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

# The line below is the answer to the prompt/query
# Wait for 25 milliseconds before displaying the next frame