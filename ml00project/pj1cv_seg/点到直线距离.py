import cv2

import numpy as np

import random

# 生成一条直线

img = np.zeros((512, 512, 3), dtype=np.uint8)

pt1 = (random.randint(0, 512), random.randint(0, 512))

pt2 = (random.randint(0, 512), random.randint(0, 512))

cv2.line(img, pt1, pt2, (0, 255, 0), 2)

# 在直线周围生成100个点

points = []

for i in range(100):

    x = random.randint(pt1[0], pt2[0])

    y = random.randint(pt1[1], pt2[1])

    points.append([x, y])

    cv2.circle(img, (x, y), 3, (255, 0, 0), -1)

# 用这些点拟合一条直线b

vx, vy, x, y = cv2.fitLine(np.array(points), cv2.DIST_L2, 0, 0.01, 0.01)

slope = vy / vx

intercept = y - slope * x

# 计算点到直线b的距离，去掉最大的20个

distances = []

for point in points:

    distance = abs(slope * point[0] - point[1] + intercept) / np.sqrt(slope ** 2 + 1)

    distances.append(distance)

sorted_indexes = np.argsort(distances)[:-20]

remaining_points = np.array(points)[sorted_indexes]

# 用剩下的80个点拟合新直线

vx, vy, x, y = cv2.fitLine(remaining_points, cv2.DIST_L2, 0, 0.01, 0.01)

slope = vy / vx

intercept = y - slope * x

# 最后显示在图像上把这些直线和点

for point in remaining_points:

    cv2.circle(img, tuple(point), 3, (0, 0, 255), -1)

y1 = int(slope * pt1[0] + intercept)

y2 = int(slope * pt2[0] + intercept)

cv2.line(img, (pt1[0], y1), (pt2[0], y2), (255, 255, 0), 2)

cv2.imshow("image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()