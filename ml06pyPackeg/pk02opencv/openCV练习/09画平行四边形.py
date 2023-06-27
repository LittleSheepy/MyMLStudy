import cv2
import numpy as np

# Load image
img = cv2.imread(r'D:\02dataset\01work\05nanjingLG\06ReJudgeBack\testSimple\img1.jpg')
#img = np.zeros((300, 409), np.uint8)

# Define points of parallelogram
pts1 = np.float32([[50, 50], [200, 50], [150, 200], [0, 200]])

# Define corresponding points in output image
pts2 = np.float32([[0, 0], [200, 0], [200, 200], [0, 200]])

# Calculate perspective transform matrix
M = cv2.getPerspectiveTransform(pts1, pts2)

# Apply perspective transform to image
dst = cv2.warpPerspective(img, M, (200, 200))

# Display result
cv2.imshow('Original Image', img)
cv2.imshow('Perspective Transform', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()