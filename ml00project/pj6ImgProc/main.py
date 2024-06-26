import os
import cv2
import os
import time
import fnmatch
# from imgproc import CKDImgProc_InspDefect_SegSepa2, CImgProc_Result  # Assuming these are available in Python
from InspDefect_SegSepa2 import CKDImgProc_InspDefect_SegSepa2
from typedef import CImgProc_Result, TImgProc_DefectParam
from PIL import Image
import numpy as np

def scan_files(path, vecfiles):
    try:
        for root, dirnames, filenames in os.walk(path):
            for filename in filenames:
                if fnmatch.fnmatch(filename, '*.jpg') or fnmatch.fnmatch(filename, '*.jpeg') or fnmatch.fnmatch(filename, '*.bmp'):
                    vecfiles.append(os.path.join(root, filename))
    except Exception as e:
        print(f"An error occurred: {e}")


class CTestDlg:
    def __init__(self, pParent=None):
        self.m_hIcon = None
        self.m_param = TImgProc_DefectParam()

        self.m_param.iKnifeNum = 8
        self.m_param.iKnifePos = [
            [2500, 2600],
            [5100, 5200],
            [7650, 7800],
            [10200, 10350],
            [12800, 12900],
            [15350, 15450],
            [17950, 18050],
            [20500, 20600]
        ]

        self.m_param.dRateWH = 1.0
        self.m_param.iThresd = 130
        self.m_param.iRetract_WB = 5
        self.m_param.iRetract_SubWB = 10
        self.m_param.iSubDefectImgSize = 256

        self.m_param.iDefectPixel_Threshold[0] = 220        # 白斑
        self.m_param.iDefectPixel_Threshold[1] = 40         # 黑斑
        self.m_param.iDefectPixel_Threshold[2] = 12         # 亚白
        self.m_param.iDefectPixel_Threshold[3] = 15         # 亚黑
        # 0 - 针孔, 1 - 油渍, 2 - 涂层脱落, 3 - 亮点， 4 - 黑点， 5 - 脏污
        self.m_param.tDefectParamArr[0].iMinWid = 1
        self.m_param.tDefectParamArr[0].iMinHei = 1
        self.m_param.tDefectParamArr[0].iMinArea = 3

        self.m_param.tDefectParamArr[1].iMinWid = 10
        self.m_param.tDefectParamArr[1].iMinHei = 10
        self.m_param.tDefectParamArr[1].iMinArea = 3000

        self.m_param.tDefectParamArr[2].iMinWid = 5
        self.m_param.tDefectParamArr[2].iMinHei = 5
        self.m_param.tDefectParamArr[2].iMinArea = 50

        self.m_param.tDefectParamArr[3].iMinWid = 5
        self.m_param.tDefectParamArr[3].iMinHei = 5
        self.m_param.tDefectParamArr[3].iMinArea = 20

        self.m_param.tDefectParamArr[4].iMinWid = 3
        self.m_param.tDefectParamArr[4].iMinHei = 3
        self.m_param.tDefectParamArr[4].iMinArea = 20

        self.m_param.tDefectParamArr[5].iMinWid = 8
        self.m_param.tDefectParamArr[5].iMinHei = 8
        self.m_param.tDefectParamArr[5].iMinArea = 300
    # def __init__(self, pParent=None):
    #     # self.m_hIcon = self.LoadIcon('IDR_MAINFRAME')
    #     self.iIdx = 0
    #     self.m_param = {
    #         'iKnifeNum': 3,
    #         'iKnifePos': [
    #             [2500, 2600],
    #             [5100, 5200],
    #             [7650, 7800],
    #             [10200, 10350],
    #             [30600, 31000],
    #             [16900, 17100],
    #             [19500, 19700],
    #             [22000, 22200],
    #             [24600, 24800],
    #             [27200, 27400]
    #         ],
    #         'dRateWH': 1.0,
    #         'iThresd': 130,
    #         'iRetract_WB': 5,
    #         'iRetract_SubWB': 10,
    #         'iSubDefectImgSize': 256,
    #         'iDefectPixel_Threshold': [220, 40, 12, 15],
    #         'tDefectParamArr': [
    #             {'iMinWid': 1, 'iMinHei': 1, 'iMinArea': 3},
    #             {'iMinWid': 10, 'iMinHei': 10, 'iMinArea': 3000},
    #             {'iMinWid': 5, 'iMinHei': 5, 'iMinArea': 50},
    #             {'iMinWid': 5, 'iMinHei': 5, 'iMinArea': 20},
    #             {'iMinWid': 3, 'iMinHei': 3, 'iMinArea': 20},
    #             {'iMinWid': 8, 'iMinHei': 8, 'iMinArea': 300}
    #         ]
    #     }
    def Thread_Inspect(self):
        vecFilePath = []
        scan_files("D:\\Test_SepaSeg\\0330漏涂", vecFilePath)

        imgProc = CKDImgProc_InspDefect_SegSepa2(23328, 2000)
        imgProc.Reset()
        imgProc.SetParam(self.m_param)

        iCurIdx = 0
        while iCurIdx < len(vecFilePath):
            pil_image = Image.open(vecFilePath[iCurIdx]).convert('L')

            # 将PIL Image对象转换为NumPy数组
            numpy_image = np.array(pil_image)

            # 将RGB图像转换为BGR图像，因为OpenCV使用BGR格式
            # img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2GRAY)
            # img = cv2.imread(vecFilePath[iCurIdx], cv2.IMREAD_GRAYSCALE)
            img = numpy_image
            iCurIdx += 1
            if img is None:
                break

            start_time = time.time()

            result = CImgProc_Result()
            imgProc.InputImage(result, img)

            elapsed_time = time.time() - start_time
            print(f"KD Save time : {elapsed_time}")

            name = ["针孔", "油渍", "涂层","亮点", "黑点", "脏污", "褶皱"]

            for idx, pBlob in enumerate(result.m_vecBlob):
                filename = f"d:\\grab\\thumb\\py_{iCurIdx - 1}_{idx}_{pBlob.iArea}_{name[pBlob.iType]}-({pBlob.iSize_Wid},{pBlob.iSize_Hei}).jpg"
                # cv2.imwrite(filename, pBlob.imgDefect)
                img = cv2.cvtColor(pBlob.imgDefect, cv2.COLOR_BGR2RGB)  # 将图像从BGR转换为RGB
                # 创建一个PIL图像并保存它
                pil_img = Image.fromarray(img)
                pil_img.save(filename)

        self.EnableOp(True)

    def EnableOp(self, value):
        pass  # Implement this function as needed

if __name__ == '__main__':
    print("")
    my_vecfiles = []
    dir_path = r"D:\02dataset\datasets_cls\train\00/"
    dlg = CTestDlg()
    dlg.Thread_Inspect()
    # try:
    #     dlg.Thread_Inspect()
    # except Exception as e:
    #     print(e)

    pass

"""
import matplotlib.pyplot as plt
plt.imshow(binImg, cmap='gray')
plt.show()

cv2.imshow("qq", roiImg)
cv2.waitKey(0)
"""