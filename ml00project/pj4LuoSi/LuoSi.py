# Step 1: Convert the color image to grayscale
import cv2
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 2: Apply adaptive weighted median filter to remove noise
filtered = cv2.medianBlur(gray, 5)

# Step 3: Enhance the image using multi-scale top-hat transform
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
tophat = cv2.morphologyEx(filtered, cv2.MORPH_TOPHAT, kernel)

# Step 4: Perform adaptive thresholding using Otsu's method
_, thresh = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Step 5: Extract the region of interest (ROI) to reduce complexity
x, y, w, h = cv2.boundingRect(thresh)
roi = thresh[y:y+h, x:x+w]

# Step 6: Apply adaptive thresholding using Canny operator to extract edge features
edges = cv2.Canny(roi, 50, 150, apertureSize=3)

# Step 7: Detect sub-pixel corners using cornerSubPix function
corners = cv2.goodFeaturesToTrack(edges, 100, 0.01, 10)
corners = cv2.cornerSubPix(edges, corners, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

# Step 8: Fit lines to the upper and lower peaks of the corners using robust linear regression
upper_peaks = corners[corners[:, :, 1] < h/2]
lower_peaks = corners[corners[:, :, 1] > h/2]
upper_line = cv2.fitLine(upper_peaks, cv2.DIST_HUBER, 0, 0.01, 0.01)
lower_line = cv2.fitLine(lower_peaks, cv2.DIST_HUBER, 0, 0.01, 0.01)

# Step 9: Calculate the distance from lower peaks to upper line and upper peaks to lower line to obtain the major diameter
distances1 = cv2.pointPolygonTest(upper_peaks, (upper_line[2], upper_line[3]), True)
distances2 = cv2.pointPolygonTest(lower_peaks, (lower_line[2], lower_line[3]), True)
major_diameter = (abs(distances1) + abs(distances2)) / (len(upper_peaks) + len(lower_peaks))

# Step 10: Calculate the distance from upper valleys to lower line and lower valleys to upper line to obtain the minor diameter
valleys1 = cv2.goodFeaturesToTrack(roi, 100, 0.01, 10, mask=255-edges)
valleys2 = cv2.goodFeaturesToTrack(roi, 100, 0.01, 10, mask=edges)
distances1 = cv2.pointPolygonTest(valleys1, (lower_line[2], lower_line[3]), True)
distances2 = cv2.pointPolygonTest(valleys2, (upper_line[2], upper_line[3]), True)
minor_diameter = (abs(distances1) + abs(distances2)) / (len(valleys1) + len(valleys2))

# Step 11: Fit lines to the midpoints of upper peaks and valleys, and lower peaks and valleys using robust linear regression
upper_midpoints = (upper_peaks + valleys2) / 2
lower_midpoints = (lower_peaks + valleys1) / 2
upper_midline = cv2.fitLine(upper_midpoints, cv2.DIST_HUBER, 0, 0.01, 0.01)
lower_midline = cv2.fitLine(lower_midpoints, cv2.DIST_HUBER, 0, 0.01, 0.01)

# Step 12: Calculate the distance between adjacent upper midpoints and adjacent lower midpoints to obtain the pitch
upper_distances = cv2.norm(upper_midpoints[1:] - upper_midpoints[:-1], axis=1)
lower_distances = cv2.norm(lower_midpoints[1:] - lower_midpoints[:-1], axis=1)
pitch = (sum(upper_distances) + sum(lower_distances)) / (len(upper_distances) + len(lower_distances))

# Step 13: Calculate the tooth angle using the arctangent of the slope of the two extended lines
u0 = corners[corners[:, :, 0] == w-1][0][0]
# d0 = corners[corners[:, :,