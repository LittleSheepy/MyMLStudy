import cv2 as cv
import os



def convertScaleAbs_dir():
    global result_folder_save
    global alpha
    for filename in os.listdir(input_folder):

        img_gray = cv.imread(input_folder + filename, cv.IMREAD_GRAYSCALE)
        max_ = img_gray.max()
        print(max_)
        scaled_img = cv.convertScaleAbs(img_gray, alpha=alpha, beta=-50)
        cv.imwrite(result_folder_save+filename, scaled_img)

def equalizeHist_dir():
    global result_folder_save
    global alpha
    for filename in os.listdir(input_folder):

        img_gray = cv.imread(input_folder + filename, cv.IMREAD_GRAYSCALE)
        equalized = cv.equalizeHist(img_gray)
        cv.imwrite(result_folder+filename, equalized)

def createCLAHE_dir():
    global result_folder_save
    global alpha
    for filename in os.listdir(input_folder):

        img_gray = cv.imread(input_folder + filename, cv.IMREAD_GRAYSCALE)
        # 应用适应性直方图均衡化
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(img_gray)

        cv.imwrite(result_folder+filename, equalized)


alpha = 0.1
if __name__ == '__main__':
    root_dir = r"D:\04DataSets\05nanjingLG\04img_scale/"
    input_folder = root_dir + 'img/'
    result_folder = root_dir + 'convertScaleAbs50/'
    result_folder_save = ""
    #createCLAHE_dir()
    for i in range(1,10,1):
        alpha = 1.0 + i/10.0
        result_folder_save = result_folder + str(i) + "/"
        os.makedirs(result_folder_save)
        convertScaleAbs_dir()
