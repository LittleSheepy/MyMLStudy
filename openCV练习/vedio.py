import cv2 as cv

def main():
    #cap = cv.VideoCapture("E:\kone_projects/KONE TM210_Cut.mp4")
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        cv.imshow("img_show", frame)
        #cv.imwrite("img", frame)
        cv.waitKey(100)

if __name__ == '__main__':
    main()