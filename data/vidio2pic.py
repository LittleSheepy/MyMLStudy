import cv2 as cv

def main():
    cap = cv.VideoCapture(r"D:\01sheepy\01work\02tongllidianti\0dataset\04lcd\WIN_20210128_16_52_58_Pro.mp4")
    n = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_num = cap.get(cv.CAP_PROP_POS_FRAMES)
        if frames_num % 1 == 0:
            cv.imwrite(r"D:\01sheepy\01work\02tongllidianti\0dataset\04lcd\img/Kone_lcd_"+str(n)+".jpg", frame)
            n = n + 1
        cv.waitKey(1)
if __name__ == '__main__':
    main()