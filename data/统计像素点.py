import os
import cv2 as cv



def main():
    root_dir = r"D:\01sheepy\01work\07xiling\00baifenbi\baifenbi_name/"
    img_dir = root_dir + r"\mask/"
    f = open(os.path.join(root_dir, "result.csv"), "w+")
    for filename in os.listdir(img_dir):
        print(filename)
        img = cv.imread(img_dir + filename,cv.IMREAD_GRAYSCALE)
        pix_all = img.shape[0] * img.shape[1]
        pix_mask = img.sum()/255
        percent = round(pix_mask/pix_all,4)*100
        print(percent)
        f.write(filename[:-4] + "," + str(percent) + "\n")
    f.close()
    pass




if __name__ == '__main__':
    main()
