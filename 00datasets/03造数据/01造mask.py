import cv2
import numpy as np

# Create a black image
img = np.zeros((3000, 4096), np.uint8)

# 画白框
#cv2.rectangle(img, (750, 260), (3280, 2750), 255, -1)
cv2.rectangle(img, (750, 2750), (3280, 260), 255, -1)
# 画黑框
cv2.rectangle(img, (850, 360), (3180, 2650), 0, -1)


# 画定位孔
cv2.rectangle(img, (850+650, 360), (850+650+70, 360+70), 100, -1)

# 画定位孔
cv2.rectangle(img, (850, 360+650), (850+70, 360+650+70), 100, -1)

# Save the image
cv2.imwrite('mask.jpg', img)
