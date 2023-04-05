import cv2
import numpy as np

# Load the image as grayscale
dir_root = r"D:\02dataset\02opencv_data/"
img_path = dir_root + 'HappyFish.jpg'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
# Threshold the image to create a binary image
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

thresh = np.int32(thresh)
# Find contours using RETR_FLOODFILL mode
contours, hierarchy = cv2.findContours(np.int32(thresh), cv2.RETR_FLOODFILL, cv2.CHAIN_APPROX_SIMPLE)

# Draw the contours on a blank image
contour_img = np.zeros_like(img)
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), 2)

# Display the image
cv2.imshow('Contours', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
