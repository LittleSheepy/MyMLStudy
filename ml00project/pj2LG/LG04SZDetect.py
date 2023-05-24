import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def pltShowCV(img, title="img"):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print('Image is single-channel (grayscale)')
    elif len(img.shape) == 3:
        print('Image is three-channel (RGB)')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title(title)
    plt.show()

def pltShowCVDot(img, point, title="img"):
    global cnt
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print('Image is single-channel (grayscale)')
    elif len(img.shape) == 3:
        print('Image is three-channel (RGB)')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img, point, 20, (255, 0, 0), -1)
    plt.imshow(img)
    plt.title(title + str(cnt))
    plt.show()
    cnt += 1


class _TKDWater_Defect:
    def __init__(self):
        self.rtRect = None
        self.iArea = 0


class CAlgo_KDWater:
    def __init__(self):
        self.m_imgTempl = None
        self.m_ptStdOffs = None

    def LoadTemplate(self):
        self.m_imgTempl = cv2.imread("Template\\nm_block.bmp", cv2.IMREAD_GRAYSCALE)
        self.m_ptStdOffs = (213, 213)
        return True

    def FindDefect(self, vecDefect, srcImgGray):
        vecDefect.clear()

        ptCent = [0, 0]
        if self._GetBestMatchPos(ptCent, srcImgGray):
            self._SearchDefect(vecDefect, ptCent, srcImgGray)

    def _SearchDefect(self, vecDefect, cent, srcImgGray):
        global filename
        vecBoxCent = [cent]
        iIndex_R = 0

        ret = True
        while iIndex_R < len(vecBoxCent):
            curCent = vecBoxCent[iIndex_R]
            iIndex_R += 1
            #pltShowCVDot(srcImgGray, curCent, "curCent")
            self._SearchDefect_Box(vecDefect, vecBoxCent, curCent, srcImgGray)
            if len(vecDefect) > 0:
                # ret = False
                # break
                pass

        imgColor = cv2.cvtColor(srcImgGray, cv2.COLOR_GRAY2BGR)

        for idx in range(len(vecDefect)):
            boxRT = vecDefect[idx].rtRect
            boxRT[0] -= 5
            boxRT[1] -= 5
            boxRT[2] += 10
            boxRT[3] += 10
            if boxRT[0] < 0:
                boxRT[0] = 0
            if boxRT[1] < 0:
                boxRT[1] = 0
            if boxRT[0] + boxRT[2] > imgColor.shape[1]:
                boxRT[2] = imgColor.shape[1] - boxRT[0]
            if boxRT[1] + boxRT[3] > imgColor.shape[0]:
                boxRT[3] = imgColor.shape[0] - boxRT[1]

            cv2.rectangle(imgColor, boxRT, (0, 0, 255), 2)
            # buf = f"{vecDefect[idx].iArea}"
            # cv2.putText(imgColor, buf, vecDefect[idx].rtRect.tl(), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        imgOut = imgColor.copy()

        iHalfW = self.m_imgTempl.shape[1] >> 1
        iHalfH = self.m_imgTempl.shape[0] >> 1
        for idx in range(len(vecBoxCent)):
            rect = (vecBoxCent[idx][0] - iHalfW, vecBoxCent[idx][1] - iHalfH, self.m_imgTempl.shape[1],
                    self.m_imgTempl.shape[0])
            cv2.rectangle(imgOut, rect, (0, 255, 0), 2)

        buf = [0] * 128
        cnt = 0
        if len(vecDefect) > 0:
            cnt += 1
            buf = f"d:\\1\\{filename}_[{len(vecDefect)}]_NG.bmp"
            cv2.imwrite(buf, imgColor)
        else:
            cnt += 1
            buf = f"d:\\1\\{filename}_OK.bmp"
            cv2.imwrite(buf, imgColor)

        # imgC2 = cv2.pyrDown(imgColor)
        # cv2.imshow("1", imgC2)
        # cv2.waitKey(0)

        if len(vecDefect) > 0:
            ret = False

        return ret

    def _addBoxPos(self, vecBoxCent, cent):
        total = len(vecBoxCent)
        ret = True
        for idx in range(total):
            dx = abs(vecBoxCent[idx][0] - cent[0])
            dy = abs(vecBoxCent[idx][1] - cent[1])

            if dx < 100 and dy < 100:
                ret = False
                break

        if ret:
            vecBoxCent.append(cent)

    #
    def _SearchDefect_Box(self, vecDefect, vecBoxCent, cent, srcImgGray):
        curRect = (
            cent[0] - (self.m_imgTempl.shape[1] >> 1), cent[1] - (self.m_imgTempl.shape[0] >> 1),
            self.m_imgTempl.shape[1],
            self.m_imgTempl.shape[0])
        imgCur = srcImgGray[curRect[1]:curRect[1] + curRect[3], curRect[0]:curRect[0] + curRect[2]]

        bNeib = [False] * 4
        imgNeib = [np.zeros((251, 247), dtype=np.uint8) for i in range(4)]
        ptNeib = [[0, 0] for i in range(4)]

        # iIndex = 0
        # ret, imgNeib[iIndex] = self._GetImage_ByBox_Left(imgNeib[iIndex], ptNeib[iIndex], srcImgGray, cent, imgCur)
        # if ret:
        #     bNeib[iIndex] = True
        #     self._addBoxPos(vecBoxCent, ptNeib[iIndex])
        #
        # iIndex = 1
        # if self._GetImage_ByBox_Right(imgNeib[iIndex], ptNeib[iIndex], srcImgGray, cent, imgCur):
        #     bNeib[iIndex] = True
        #     self._addBoxPos(vecBoxCent, ptNeib[iIndex])
        #
        # iIndex = 2
        # if self._GetImage_ByBox_Up(imgNeib[iIndex], ptNeib[iIndex], srcImgGray, cent, imgCur):
        #     bNeib[iIndex] = True
        #     self._addBoxPos(vecBoxCent, ptNeib[iIndex])
        #
        # iIndex = 3
        # if self._GetImage_ByBox_Down(imgNeib[iIndex], ptNeib[iIndex], srcImgGray, cent, imgCur):
        #     bNeib[iIndex] = True
        #     self._addBoxPos(vecBoxCent, ptNeib[iIndex])
        # for idx in range(4):
        #     if not bNeib[idx]:
        #         imgNeib[idx] = imgCur.copy()
        #
        funcList = [self._GetImage_ByBox_Left, self._GetImage_ByBox_Right, self._GetImage_ByBox_Up, self._GetImage_ByBox_Down]
        for idx in range(4):
            ret, imgNeib[idx] = funcList[idx](imgNeib[idx], ptNeib[idx], srcImgGray, cent, imgCur)
            if ret:
                bNeib[idx] = True
                self._addBoxPos(vecBoxCent, ptNeib[idx])
            else:
                imgNeib[idx] = imgCur.copy()

        DEF_THRESD_AREA = 100
        DEF_THRESD_NOR = 50
        DEF_THRESD_DIFF = 60

        diff = [np.zeros((1, 1), dtype=np.uint8)] * 4
        # for i in range(4):
        #     imgNeib[i] = np.zeros((251, 247), dtype=np.uint8)
        for idx in range(4):
            diff[idx] = cv2.absdiff(imgNeib[idx], imgCur)
            _, diff[idx] = cv2.threshold(diff[idx], DEF_THRESD_DIFF, 1, cv2.THRESH_BINARY)

        diff[0] += diff[1] + diff[2] + diff[3]

        imgBin = cv2.threshold(diff[0], 1, 255, cv2.THRESH_BINARY)[1]

        imgBaseBin = cv2.threshold(imgCur, DEF_THRESD_NOR, 255, cv2.THRESH_BINARY_INV)[1]
        imgBin &= imgBaseBin

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        imgBin = cv2.erode(imgBin, kernel)
        imgBin = cv2.dilate(imgBin, kernel)

        contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        #vecDefect = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > DEF_THRESD_AREA:
                defect = _TKDWater_Defect()
                defect.iArea = area
                defect.rtRect = list(cv2.boundingRect(cnt))
                defect.rtRect[0] += curRect[0]
                defect.rtRect[1] += curRect[1]
                vecDefect.append(defect)

    # def _GetDefect_ByImgDiff(self, defect, imgAux, imgBase):
    #     DEF_THRESD_NOR = 60
    #     DEF_THRESD_DIFF = 80
    #     DEF_THRESD_ABNOR = 30
    #     DEF_THRESD_AREA = 300
    #
    #     img0 = cv2.GaussianBlur(imgAux, (5, 5), 1.0)
    #     img1 = cv2.GaussianBlur(imgBase, (5, 5), 1.0)
    #
    #     img0 = cv2.threshold(imgBase, DEF_THRESD_ABNOR, 255, cv2.THRESH_BINARY_INV)[1]
    #     img1 = cv2.threshold(imgAux - imgBase, DEF_THRESD_DIFF, 255, cv2.THRESH_BINARY)[1]
    #     imgBin = cv2.bitwise_and(img1, img0)
    #
    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    #     img1 = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel)
    #
    #     contours, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     ret = False
    #
    #     for cnt in contours:
    #         area = cv2.contourArea(cnt)
    #         if area > DEF_THRESD_AREA:
    #             defect.iArea = area
    #             defect.rtRect = cv2.boundingRect(cnt)
    #             ret = True
    #
    #     return ret

    def _GetImage_ByBox_Left(self, outImg, outCent, srcImg, cent, curImgTempl):
        iHalfWid = (self.m_imgTempl.shape[1] >> 1) + 30
        iHalfHei = (self.m_imgTempl.shape[0] >> 1) + 30

        roiRect = (cent[0] - iHalfWid - self.m_ptStdOffs[0], cent[1] - iHalfHei, iHalfWid << 1, iHalfHei << 1)
        if roiRect[0] < 0 or roiRect[1] < 0 or roiRect[0] + roiRect[2] > srcImg.shape[1] or roiRect[1] + roiRect[3] > \
                srcImg.shape[0]:
            return False, None

        roiImg = srcImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]
        ret, outImg = self._GetMaxMatch(outImg, outCent, roiImg, curImgTempl)
        if ret:
            outCent[0] += roiRect[0]
            outCent[1] += roiRect[1]

        return ret, outImg

    def _GetImage_ByBox_Right(self, outImg, outCent, srcImg, cent, curImgTempl):
        iHalfWid = (self.m_imgTempl.shape[1] >> 1) + 30
        iHalfHei = (self.m_imgTempl.shape[0] >> 1) + 30

        roiRect = (cent[0] - iHalfWid + self.m_ptStdOffs[0], cent[1] - iHalfHei, iHalfWid << 1, iHalfHei << 1)
        if (roiRect[0] < 0) or (roiRect[1] < 0) or (roiRect[0] + roiRect[2] > srcImg.shape[1]) or (
                roiRect[1] + roiRect[3] > srcImg.shape[0]):
            return False, None

        roiImg = srcImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]
        ret, outImg = self._GetMaxMatch(outImg, outCent, roiImg, curImgTempl)
        if ret:
            outCent[0] += roiRect[0]
            outCent[1] += roiRect[1]

        return ret, outImg

    def _GetImage_ByBox_Up(self, outImg, outCent, srcImg, cent, curImgTempl):
        iHalfWid = (self.m_imgTempl.shape[1] >> 1) + 30
        iHalfHei = (self.m_imgTempl.shape[0] >> 1) + 30

        roiRect = (cent[0] - iHalfWid, cent[1] - iHalfHei - self.m_ptStdOffs[1], iHalfWid << 1, iHalfHei << 1)
        if (roiRect[0] < 0) or (roiRect[1] < 0) or (roiRect[0] + roiRect[2] > srcImg.shape[1]) or (
                roiRect[1] + roiRect[3] > srcImg.shape[0]):
            return False, None

        roiImg = srcImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]
        ret, outImg = self._GetMaxMatch(outImg, outCent, roiImg, curImgTempl)
        if ret:
            outCent[0] += roiRect[0]
            outCent[1] += roiRect[1]

        return ret, outImg

    def _GetImage_ByBox_Down(self, outImg, outCent, srcImg, cent, curImgTempl):
        iHalfWid = (self.m_imgTempl.shape[1] >> 1) + 30
        iHalfHei = (self.m_imgTempl.shape[0] >> 1) + 30

        roiRect = (cent[0] - iHalfWid, cent[1] - iHalfHei + self.m_ptStdOffs[1], iHalfWid << 1, iHalfHei << 1)
        if roiRect[0] < 0 or roiRect[1] < 0 or roiRect[0] + roiRect[2] > srcImg.shape[1] or roiRect[1] + roiRect[3] > \
                srcImg.shape[0]:
            return False, None

        roiImg = srcImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]
        ret, outImg = self._GetMaxMatch(outImg, outCent, roiImg, curImgTempl)
        if ret:
            outCent[0] += roiRect[0]
            outCent[1] += roiRect[1]

        return ret, outImg

    def _GetMaxMatch(self, outImg, outCent, roiImg, imgTempl):
        result = cv2.matchTemplate(roiImg, imgTempl, cv2.TM_CCOEFF_NORMED)
        result = np.clip(result, 0, np.inf)
        result = result * 255
        #result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        dMax = 0
        _, dMax, _, maxLoc = cv2.minMaxLoc(result)
        iVal = dMax
        if iVal < 100:
            return False, None
        iSumX = 0
        iSumY = 0
        cnt = 0
        for row in range(result.shape[0]):
            pLine = result[row, :]
            for col in range(result.shape[1]):
                if pLine[col] >= iVal:
                    iSumX += col
                    iSumY += row
                    cnt += 1

        if cnt <= 0:
            return False

        outCent[0] = iSumX // cnt
        outCent[1] = iSumY // cnt

        roiRect = (outCent[0], outCent[1], imgTempl.shape[1], imgTempl.shape[0])    # row:行数 高度 251  cols: 宽度 247
        outImg = roiImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]

        outCent[0] += imgTempl.shape[1] // 2
        outCent[1] += imgTempl.shape[0] // 2

        return True, outImg

    def _GetMaxMatch_jianhua(self, outImg, outCent, roiImg, imgTempl):
        result = cv2.matchTemplate(roiImg, imgTempl, cv2.TM_CCOEFF_NORMED)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        dMax = 0
        _, dMax, _, maxLoc = cv2.minMaxLoc(result)
        if dMax < 100:
            return False

        outCent[0] = maxLoc[0] + imgTempl.shape[1] // 2
        outCent[1] = maxLoc[1] + imgTempl.shape[0] // 2

        roiRect = (maxLoc[0], maxLoc[1], imgTempl.shape[1], imgTempl.shape[0])
        outImg = roiImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]

        return True

    def _GetBestMatchPos(self, ptCent, srcImgGray):
        result = cv2.matchTemplate(srcImgGray, self.m_imgTempl, cv2.TM_CCOEFF_NORMED)
        # result = result * 255
        # result = np.uint8(result)
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        dMax = 0
        _, dMax, _, _ = cv2.minMaxLoc(result)

        dMax -= 1.0
        if dMax < 100:
            dMax = 100

        _, imgBin = cv2.threshold(result, dMax, 255, cv2.THRESH_BINARY)

        vecCont, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        vecRect = []
        iMaxV = 0
        iMaxIdx = -1
        for idx in range(len(vecCont)):
            area = len(vecCont[idx])
            if area > iMaxV:
                iMaxV = area
                iMaxIdx = idx

        ret = False

        if iMaxIdx >= 0:
            rect = cv2.boundingRect(vecCont[iMaxIdx])

            ptCent[0] = rect[0] + (rect[2] >> 1) + (self.m_imgTempl.shape[1] >> 1)
            ptCent[1] = rect[1] + (rect[3] >> 1) + (self.m_imgTempl.shape[0] >> 1)

            ret = True

        return ret


def test_water():
    global filename
    water = CAlgo_KDWater()
    water.LoadTemplate()

    vecFiles = []
    _scan_files("D:\\LGCBTray\\Test_Image\\NM", vecFiles)

    vecDefect = []
    for idx in range(len(vecFiles)):
        img = cv2.imread(vecFiles[idx], cv2.IMREAD_GRAYSCALE)
        pltShowCV(img, "OriImg")
        file_name = os.path.basename(vecFiles[idx])
        filename = os.path.splitext(file_name)[0]
        water.FindDefect(vecDefect, img)


def _scan_files(directory, vecFiles):
    for file in os.listdir(directory):
        if file.endswith(".jpg") or file.endswith(".png"):
            vecFiles.append(os.path.join(directory, file))

filename = ""
cnt = 1
if __name__ == '__main__':
    test_water()

"""
    cv2.imwrite("test.jpg", result)
"""
