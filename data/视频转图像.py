"""" 将视频转换成图片 path: 视频路径 """
import os
import cv2
import numpy as np

def cv2_img():
    path=r'D:\01sheepy\01work\02tongllidianti\0dataset\01val\tongli_mp4_0115\1.mp4'
    out_dir = r"D:\01sheepy\01work\02tongllidianti\0dataset\01val\val_0115/"
    cap = cv2.VideoCapture(path)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    num=0
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        # print(suc)
        # params = []
        # params.append(2)  # params.append(1)
        # cv2.imshow("img",frame)
        # cv2.waitKey(1)
        pic_path = "val_kone01" + "_" + str(num)
        if frame_count % 5 == 0:
            cv2.imwrite(out_dir + '/%s.jpg' % pic_path, frame)
            num+=1
        # cv2.destroyAllWindows()
        # if frame_count % 20==0:
        #     pic_path="kone"+"_"+str(frame_count)
        #     cv2.imwrite('video\\kone\\%s.jpg' % pic_path, frame)

    # cap.release()
    # print('unlock movie: ', frame_count)
def video2img(videoPath, out_dir, num_start):
    cap = cv2.VideoCapture(videoPath)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    num=num_start
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        #np_arrpy = np.identity()
        if not suc:
            break
        frame = np.rot90(frame,3)
        pic_path = "lab_115" + "_" + str(num)
        if frame_count % 1 == 0:
            cv2.imwrite(out_dir + '/%s.jpg' % pic_path, frame)
            num+=1
    return num

def dir2img(video_dir, out_dir):
    num = 0
    for file in os.listdir(video_dir):
        print(file)
        num = video2img(video_dir + file, out_dir, num)

if __name__ == '__main__':
    video_dir = r'D:\01sheepy\01work\02tongllidianti\0dataset\03\AVI/'
    out_dir = r"D:\01sheepy\01work\02tongllidianti\0dataset\03\img/"
    dir2img(video_dir, out_dir)