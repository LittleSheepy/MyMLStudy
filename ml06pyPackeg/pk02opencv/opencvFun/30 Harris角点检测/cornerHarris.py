import cv2 as cv

def nothing(x):
    pass

cv.namedWindow('cornerHarris',cv.WINDOW_NORMAL)
cv.createTrackbar('blockSize','cornerHarris',5,10, nothing)
cv.createTrackbar('ksize','cornerHarris',5,10, nothing)
cv.createTrackbar('k','cornerHarris',4,6, nothing)
cv.createTrackbar('per','cornerHarris',10,30, nothing)


while True:
    blockSize = cv.getTrackbarPos('blockSize','cornerHarris')
    ksize = cv.getTrackbarPos('ksize','cornerHarris')
    k = cv.getTrackbarPos('k','cornerHarris')
    per = cv.getTrackbarPos('per','cornerHarris')
    if blockSize == -1: break
    if blockSize == 0:
        cv.setTrackbarPos('blockSize','cornerHarris',1)
        continue
    if ksize%2 == 0:
        cv.waitKey(1)
        continue

    img = cv.imread('chessboard-3.png', cv.IMREAD_COLOR)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, blockSize, ksize, 0.01 * k)
    #dst = cv.dilate(dst, None)
    img[dst > 0.01 * per * dst.max()] = [0, 0, 255]
    img[dst < 0.01 * per * dst.min()] = [0, 255, 0]

    cv.imshow('cornerHarris',img)
    cv.waitKey(1)

cv.destroyAllWindows()