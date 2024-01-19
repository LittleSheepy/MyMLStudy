import cv2
import numpy as np
from InspDefect_GetColEdge import CKDImgProc_GetBlackColEdge, CKDImgProc_GetWhiteColEdge
from typedef import *
from GetBlob import *

class CKDImgProc_InspDefect_SegSepa2:
    def __init__(self, iCol, iRow):
        self.m_iKnifePos_Cur = [[0 for _ in range(2)] for _ in range(32)]
        self.m_param = TImgProc_DefectParam()
        self.m_curImgSize = (iRow, iCol)
        self.m_curRecvImgBuf = np.zeros((iRow * 3, iCol), dtype=np.uint8)
        self.m_curBinImgBuf = np.zeros((iRow * 3, iCol), dtype=np.uint8)
        self.Reset()
    def SetParam(self, param):
        self.m_param = param

    def Reset(self):
        self.m_iKnifePos_Cur = [[0 for _ in range(2)] for _ in range(32)]
        self.m_curRecvImgBuf.fill(0)
        self.m_curBinImgBuf.fill(0)
        self.m_icurMeanLineCnt = 0
        self.m_curLineMean = []

    def GetMembranceMeanGrayValue(self):
        return self.m_curMembranceMeanGrayValue

    def GetFullImage_Cur(self, outImg):
        sz = (self.m_curRecvImgBuf.shape[1] // 2, self.m_curRecvImgBuf.shape[0] // 2)
        outImg = cv2.resize(self.m_curRecvImgBuf, sz)

    def InputImage(self, result, imgSrc):
        if imgSrc.shape != self.m_curImgSize:
            return
        print(">1 _move_imagedata")
        self._move_imagedata(imgSrc)
        print("<1 _move_imagedata")
        print(">2 _position_segment")
        self._position_segment(imgSrc)
        print("<2 _position_segment")
        print(">3 _threshold_defect_dot")
        self._threshold_defect_dot(imgSrc)
        print("<3 _threshold_defect_dot")
        print(">4 _get_defect")
        self._get_defect(result.m_vecBlob)
        print("<4 _get_defect")

    def _get_defect(self, vecBlob):
        vecBlobArr = [[] for _ in range(32)]
        for idx in range(self.m_param.iKnifeNum - 1):
            self._get_defect_segment(idx, vecBlobArr[idx])
        vecBlob_Tmp = []
        for idx in range(self.m_param.iKnifeNum - 1):
            vecBlob_Tmp.extend(vecBlobArr[idx])
        vecBlob.clear()
        iCnt_Big = 0
        for idx in range(len(vecBlob_Tmp)):
            if vecBlob_Tmp[idx].iType == 5:
                if vecBlob_Tmp[idx].iSize_Hei > 500 or vecBlob_Tmp[idx].iSize_Wid > 500:
                    if iCnt_Big < 1:
                        iCnt_Big += 1
                    else:
                        continue
            vecBlob.append(vecBlob_Tmp[idx])
        self._get_defect_image(vecBlob)

    def _get_defect_image(self, vecBlob):
        for idx in range(len(vecBlob)):
            pBlob = vecBlob[idx]
            rect = (pBlob.iPos_X, pBlob.iPos_Y, pBlob.iSize_Wid, pBlob.iSize_Hei)
            dRateWH = self.m_param.dRateWH
            if dRateWH < 0.1:
                dRateWH = 1.0
            iStdHei = int(self.m_param.iSubDefectImgSize * self.m_param.dRateWH)
            iStdWid = int(self.m_param.iSubDefectImgSize)

            cent = (rect[0] + (rect[2] >> 1), rect[1] + (rect[3] >> 1))
            wid = int(rect[2] + 20)
            hei = int(rect[3] + 20 * dRateWH)

            if (wid < iStdWid) and (hei < iStdHei):
                wid = iStdWid
                hei = iStdHei
            else:
                rate_w = wid / iStdWid
                rate_h = hei / iStdHei
                if rate_w > rate_h:
                    hei = int(rate_w * iStdHei)
                else:
                    wid = int(rate_h * iStdWid)
            if wid > 2000:
                wid = 2000
            if hei > 2000 * dRateWH:
                hei = int(2000 * dRateWH)
            rtDefect_src = (cent[0] - (wid >> 1), cent[1] - (hei >> 1), wid, hei)   # (3672, 2506, 348, 348)
            imgDefect = None
            if (rtDefect_src[0] >= 0) and (rtDefect_src[1] >= 0) and (rtDefect_src[0] + rtDefect_src[2] <= self.m_curRecvImgBuf.shape[1]) and (rtDefect_src[1] + rtDefect_src[3] <= self.m_curRecvImgBuf.shape[0]):
                imgDefect = self.m_curRecvImgBuf[rtDefect_src[1]:rtDefect_src[1]+rtDefect_src[3], rtDefect_src[0]:rtDefect_src[0]+rtDefect_src[2]].copy()
            else:
                imgDefect = np.zeros((rtDefect_src[3], rtDefect_src[2]), dtype=np.uint8)
                srcRect = list(rtDefect_src)
                dstRect = [0, 0, rtDefect_src[2], rtDefect_src[3]]
                if srcRect[0] < 0:
                    offs_x = -srcRect[0]
                    srcRect[0] = 0
                    srcRect[2] -= offs_x
                    dstRect[0] = offs_x
                    dstRect[2] = srcRect[2]
                elif srcRect[0] + srcRect[2] > self.m_curRecvImgBuf.shape[1]:
                    ilen = self.m_curRecvImgBuf.shape[1] - srcRect[0]
                    srcRect[2] = ilen
                    dstRect[0] = 0
                    dstRect[2] = ilen
                self.m_curRecvImgBuf[srcRect[1]:srcRect[1]+srcRect[3], srcRect[0]:srcRect[0]+srcRect[2]].copyTo(imgDefect[dstRect[1]:dstRect[1]+dstRect[3], dstRect[0]:dstRect[0]+dstRect[2]])
            maxwh = max(wid, hei)
            pBlob.imgDefect = cv2.resize(imgDefect, (maxwh, maxwh))
            pBlob.iPos_X += pBlob.iSize_Wid >> 1
            pBlob.iPos_Y += pBlob.iSize_Hei >> 1
            pBlob.iPos_Y -= self.m_curImgSize[0] << 1

    def _get_defect_segment(self, iSegIdx, vecBlob_Out):
        print("> <func> _get_defect_segment")
        iStX  = int(self.m_iKnifePos_Cur[iSegIdx][1])
        iEndX = int(self.m_iKnifePos_Cur[iSegIdx+1][0])
        if iEndX <= iStX:
            iEndX = iStX
        roiRect = (iStX, 0, iEndX - iStX + 1, self.m_curImgSize[0] * 3)
        binImg = self.m_curBinImgBuf[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]]
        grayImg = self.m_curRecvImgBuf[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]]
        vecColMean = self.m_curLineMean[roiRect[0]:roiRect[0]+roiRect[2]]
        self._get_blob(iSegIdx, vecBlob_Out, binImg, grayImg, vecColMean)
        for idx in range(len(vecBlob_Out)):
            pBlob = vecBlob_Out[idx]
            pBlob.iPos_X += roiRect[0]
            pBlob.iPos_Y += roiRect[1]
        print("< <func> _get_defect_segment")

    def _merge_defect_segment(self, vecBlob, vecSubBlob):
        vecTmp = []
        total = len(vecBlob)
        total_sub = len(vecSubBlob)
        for col in range(total_sub):
            rect_cur = (vecSubBlob[col].iPos_X, vecSubBlob[col].iPos_Y, vecSubBlob[col].iSize_Wid, vecSubBlob[col].iSize_Hei)
            flag = False
            for idx in range(total):
                if vecSubBlob[col].iType == vecBlob[idx].iType:
                    rect_s = (vecBlob[idx].iPos_X, vecBlob[idx].iPos_Y, vecBlob[idx].iSize_Wid, vecBlob[idx].iSize_Hei)
                    if self._get_distance(rect_cur, rect_s) < 128:
                        rect_s = (rect_s[0], rect_s[1], rect_cur[2] + rect_cur[0] - rect_s[0], rect_cur[3] + rect_cur[1] - rect_s[1])
                        vecBlob[idx].iArea += vecSubBlob[col].iArea
                        vecBlob[idx].iPos_X = rect_s[0]
                        vecBlob[idx].iPos_Y = rect_s[1]
                        vecBlob[idx].iSize_Wid = rect_s[2]
                        vecBlob[idx].iSize_Hei = rect_s[3]
                        flag = True
                        break
            if not flag:
                vecTmp.append(vecSubBlob[col])
        if len(vecTmp) > 0:
            vecBlob.extend(vecTmp)

    def _get_blob(self, iSegIdx, vecBlob_Out, binImg, grayImg, vecColMean):
        print("> <func> _get_blob")
        vecBlob_Out.clear()
        # Assuming that you have a function named GetBlobList_BinaryImage_forAll
        blobSet_0 = CAM_BlobSet()
        CAMALG_GetBlobList.GetBlobList_BinaryImage_forAll(blobSet_0, binImg, 3)
        total_0 = blobSet_0.GetTotal()
        if total_0 <= 0:
            return
        blobSet_1 = CAM_BlobSet()
        for idx in range(total_0):
            # Assuming that you have a function named InputMergeBlob
            blobSet_1.InputMergeBlob(blobSet_0.GetElement(idx), 64)
        total_1 = blobSet_1.GetTotal()
        if total_1 <= 0:
            return
        for idx in range(total_1):
            pBlob = blobSet_1.GetElement(idx)
            rect = pBlob.GetRect()
            cy = rect.y + (rect.height >> 1)
            if (cy >= self.m_curImgSize[0]) and (cy < (self.m_curImgSize[0] << 1)):
                blob = CImgProc_Blob()
                if self._blob_classify(blob, pBlob, grayImg, vecColMean):
                    blob.iSegIdx = iSegIdx
                    blob.iPos_X = rect.x
                    blob.iPos_Y = rect.y
                    blob.iSize_Hei = rect.height
                    blob.iSize_Wid = rect.width
                    rect = pBlob.GetRect()
                    blob.rcRect = (rect.x,rect.y,rect.width,rect.height)
                    self._blob_getPixelValue(blob.iPixelValue, pBlob, grayImg)
                    vecBlob_Out.append(blob)
        print("< <func> _get_blob")

    def _blob_classify(self, defectBlob, pBlob, srcImg, vecColMean):
        assert srcImg.shape[1] == len(vecColMean)

        amRect = pBlob.GetRect()

        pBlob.CheckLikeLine()

        iThresd_W = self.m_param.iDefectPixel_Threshold[0]
        iThresd_B = self.m_param.iDefectPixel_Threshold[1]
        iThresd_SW = self.m_param.iDefectPixel_Threshold[2]
        iThresd_SB = self.m_param.iDefectPixel_Threshold[3]
        iCnt_W = 0
        iCnt_SW = 0
        iCnt_SB = 0
        iCnt_B = 0
        iCnt_250 = 0

        total = pBlob.GetTotal()

        for idx in range(total):
            pLine = pBlob.GetElement(idx)

            iY = pLine.y
            pDataLine = srcImg[iY]
            iStx = pLine.xSt
            iEndx = pLine.xEnd
            iEndx_Loop = pLine.xEnd + 1

            for col in range(iStx, iEndx_Loop):
                val = pDataLine[col]
                meanVal = vecColMean[col]

                if val >= iThresd_W:        # 白
                    if val >= 250:
                        iCnt_250 += 1
                    iCnt_W += 1
                elif val >= meanVal + iThresd_SW:   # 亚白
                    iCnt_SW += 1
                elif val <= iThresd_B:
                    iCnt_B += 1
                elif val <= meanVal - iThresd_SB:   # 亚黑
                    iCnt_SB += 1

        # <2. 瑕疵分类>
        # 0 - 针孔, 1 - 油渍, 2 - 涂层脱落, 3 - 亮点, 4 - 黑点, 5 - 脏污, 6 - 褶皱
        # 0 - 针孔
        if _InspSeg_CheckDefectData(iCnt_W, amRect.width, amRect.height, self.m_param.tDefectParamArr[0]):
            if iCnt_W > self.m_param.tDefectParamArr[1].iMinArea:
                if iCnt_SW <= iCnt_W / 20:  # 白像素点数是亚白的20倍以上
                    defectBlob.iType = 0
                    defectBlob.iArea = iCnt_W + iCnt_SW
                    defectBlob.iSize_Wid = amRect.width
                    defectBlob.iSize_Hei = amRect.height
                    defectBlob.iPos_X = amRect.x
                    defectBlob.iPos_Y = amRect.y
                    return True
            else:  # 白像素个数面积达标，且未达到油渍面积
                if pBlob.m_iNeibBlobCnt < 5:  # 相邻斑点块数量不超过2个
                    if iCnt_W + iCnt_SW <= 1000:
                        defectBlob.iType = 0
                        defectBlob.iArea = iCnt_W + iCnt_SW
                        defectBlob.iSize_Wid = amRect.width
                        defectBlob.iSize_Hei = amRect.height
                        defectBlob.iPos_X = amRect.x
                        defectBlob.iPos_Y = amRect.y
                        return True
                    elif iCnt_SW <= iCnt_W * 3 / 2:  # 4
                        defectBlob.iType = 0
                        defectBlob.iArea = iCnt_W
                        defectBlob.iSize_Wid = amRect.width
                        defectBlob.iSize_Hei = amRect.height
                        defectBlob.iPos_X = amRect.x
                        defectBlob.iPos_Y = amRect.y
                        return True

        bDefect = [False] * 5
        iArea = [iCnt_W + iCnt_SW] * 3 + [iCnt_B, iCnt_SB + iCnt_B]

        for i in range(5):
            bDefect[i] = _InspSeg_CheckDefectData(iArea[i], amRect.width, amRect.height,
                                                       self.m_param.tDefectParamArr[i + 1])

        if bDefect[0]:
            if iCnt_SW * 3 < iCnt_W:
                defectBlob.iArea = iArea[0]
                defectBlob.iSize_Wid = amRect.width
                defectBlob.iSize_Hei = amRect.height
                defectBlob.iPos_X = amRect.x
                defectBlob.iPos_Y = amRect.y
                defectBlob.iType = 1
                return True

        if bDefect[1] and bDefect[2]:
            bDefect[2] = False

        if bDefect[3] and bDefect[4]:
            if iCnt_B >= iCnt_SB:
                bDefect[3] = True
                bDefect[4] = False
            else:
                bDefect[3] = False
                bDefect[4] = True

        iMaxIdx = -1
        iMaxArea = 0
        for idx in range(1, 5):
            if bDefect[idx]:
                if iArea[idx] > iMaxArea:
                    iMaxArea = iArea[idx]
                    iMaxIdx = idx

        if iMaxIdx >= 0:
            defectBlob.iArea = iArea[iMaxIdx]
            defectBlob.iSize_Wid = amRect.width
            defectBlob.iSize_Hei = amRect.height
            defectBlob.iPos_X = amRect.x
            defectBlob.iPos_Y = amRect.y
            defectBlob.iType = iMaxIdx + 1

            if defectBlob.iType == 5:
                if pBlob.m_bIsLikeLine:
                    defectBlob.iType = 6
            return True

        return False

    def _blob_getPixelValue(self, iPxVal:list, pBlob, grayImg):
        # 这里假设CAM_Blob是已经定义好的类，并且有相应的属性和方法
        # 你需要根据实际情况来实现这些类的细节
        # 同样，这里的逻辑也需要根据实际的图像处理逻辑来实现，以下是一个示例框架

        iTotal = pBlob.GetTotal()
        minV = 255
        maxV = 0
        sumV = 0
        cnt = 0
        for idx in range(iTotal):
            pLine = pBlob.GetElement(idx)
            pDataLine = grayImg[pLine.y]
            for col in range(pLine.xSt, pLine.xEnd):
                val = pDataLine[col]
                if val < minV:
                    minV = val
                if val > maxV:
                    maxV = val

                sumV += val
                cnt += 1

        if cnt < 1:
            cnt = 1

        iPxVal.extend([minV, maxV, sumV // cnt])

        return True


    def _threshold_defect_dot(self, imgSrc):
        roiRect = (0, self.m_curImgSize[0] * 2, self.m_curImgSize[1], self.m_curImgSize[0])
        curSrc = self.m_curRecvImgBuf[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]   # (2000, 23328)

        # cv2.GaussianBlur(imgSrc, curSrc, (3, 3), 0.6)
        np.copyto(curSrc, imgSrc)
        size = roiRect[2] * roiRect[3]

        imgMean = np.empty_like(curSrc)
        self._getLineMean(imgMean, curSrc)

        curBin = self.m_curBinImgBuf[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]]
        curBin.fill(0)

        for idx in range(self.m_param.iKnifeNum - 1):
            self._threshold_defect_dot_Segment(idx, curSrc, imgMean, curBin)

    def _threshold_defect_dot_Segment(self, iSegIdx, imgSrc, imgMean, imgBin):
        iStX = int(self.m_iKnifePos_Cur[iSegIdx][1] + self.m_param.iRetract_WB)
        iEndX = int(self.m_iKnifePos_Cur[iSegIdx + 1][0] - self.m_param.iRetract_WB)
        if iEndX <= iStX:
            iEndX = iStX
        roiRect_WB = (iStX, 0, iEndX - iStX + 1, self.m_curImgSize[0])

        iStX = int(self.m_iKnifePos_Cur[iSegIdx][1] + self.m_param.iRetract_SubWB)
        iEndX = int(self.m_iKnifePos_Cur[iSegIdx + 1][0] - self.m_param.iRetract_SubWB)
        if iEndX <= iStX:
            iEndX = iStX
        roiRect_SubWB = (iStX, 0, iEndX - iStX + 1, self.m_curImgSize[0])

        iThresd_W = self.m_param.iDefectPixel_Threshold[0]
        iThresd_B = self.m_param.iDefectPixel_Threshold[1]
        iThresd_SW = self.m_param.iDefectPixel_Threshold[2]
        iThresd_SB = self.m_param.iDefectPixel_Threshold[3]

        imgSrc_Roi = imgSrc[roiRect_WB[1]:roiRect_WB[1] + roiRect_WB[3], roiRect_WB[0]:roiRect_WB[0] + roiRect_WB[2]]
        imgBin_Roi = imgBin[roiRect_WB[1]:roiRect_WB[1] + roiRect_WB[3], roiRect_WB[0]:roiRect_WB[0] + roiRect_WB[2]]
        imgMean_Roi = imgMean[roiRect_WB[1]:roiRect_WB[1] + roiRect_WB[3], roiRect_WB[0]:roiRect_WB[0] + roiRect_WB[2]]

        iLineStepArr = [0] * self.m_curImgSize[0]
        for row in range(1, self.m_curImgSize[0]):
            iLineStepArr[row] = iLineStepArr[row - 1] + imgSrc_Roi.strides[0]

        for row in range(imgBin_Roi.shape[0]):
            pLine_Src = imgSrc_Roi[row]
            pLine_Bin = imgBin_Roi[row]

            for col in range(imgBin_Roi.shape[1]):
                val_src = pLine_Src[col]
                if val_src > iThresd_W or val_src < iThresd_B:
                    pLine_Bin[col] = 255

        imgSrc_Roi = imgSrc[roiRect_SubWB[1]:roiRect_SubWB[1] + roiRect_SubWB[3], roiRect_SubWB[0]:roiRect_SubWB[0] + roiRect_SubWB[2]]
        imgBin_Roi = imgBin[roiRect_SubWB[1]:roiRect_SubWB[1] + roiRect_SubWB[3], roiRect_SubWB[0]:roiRect_SubWB[0] + roiRect_SubWB[2]]
        imgMean_Roi = imgMean[roiRect_SubWB[1]:roiRect_SubWB[1] + roiRect_SubWB[3], roiRect_SubWB[0]:roiRect_SubWB[0] + roiRect_SubWB[2]]

        for row in range(imgBin_Roi.shape[0]):
            pLine_Src = imgSrc_Roi[row]
            pLine_Bin = imgBin_Roi[row]
            pLine_Mean = imgMean_Roi[row]

            for col in range(imgBin_Roi.shape[1]):
                val_src = pLine_Src[col]
                val_mean = pLine_Mean[col]
                if val_src > val_mean + iThresd_SW:
                    pLine_Bin[col] = 255
                elif val_src < val_mean - iThresd_SB:
                    pLine_Bin[col] = 255
        pass
    def _getCurLineMean(self, imgMeanLine, imgSrc):
        vecLineSum = [0] * imgSrc.shape[1]
        vecLineCnt = [0] * imgSrc.shape[1]
        vecLineMean = [0] * imgSrc.shape[1]

        iMinPxValue = self.m_param.iDefectPixel_Threshold[1]
        iMaxPxValue = self.m_param.iDefectPixel_Threshold[0]
        iLastIdx = len(self.m_param.iKnifeNum) - 1
        iStartPos = self.m_iKnifePos_Cur[0][0]
        iEndPos = self.m_iKnifePos_Cur[iLastIdx][0]
        for row in range(0, imgSrc.shape[0], 32):
            pLine = imgSrc[row]
            for col in range(iStartPos, iEndPos):
                val = pLine[col]
                if iMinPxValue <= val < iMaxPxValue:
                    vecLineSum[col - iStartPos] += val
                    vecLineCnt[col - iStartPos] += 1

        iLastVal = 140
        for col in range(iStartPos, iEndPos):
            val = 0
            if vecLineCnt[col - iStartPos] > 0:
                val = (vecLineSum[col] - iStartPos) // vecLineCnt[col - iStartPos]
                iLastVal = val
            else:
                val = iLastVal
            if val > 255:
                val = 255

            vecLineMean[col - iStartPos] = val

        iTotalSum = sum(vecLineMean)
        self.m_curMembranceMeanGrayValue = iTotalSum // (iEndPos - iStartPos)

    def _getLineMean(self, imgMeanLine, imgSrc):
        vecLineSum = [0] * imgSrc.shape[1]
        vecLineCnt = [0] * imgSrc.shape[1]
        vecLineMean = [0] * imgSrc.shape[1]

        iMinPxValue = self.m_param.iDefectPixel_Threshold[1]
        iMaxPxValue = self.m_param.iDefectPixel_Threshold[0]

        for row in range(0, imgSrc.shape[0], 32):
            pLine = imgSrc[row]
            for col in range(imgSrc.shape[1]):
                val = pLine[col]
                if iMinPxValue <= val < iMaxPxValue:
                    vecLineSum[col] += val
                    vecLineCnt[col] += 1

        iLastVal = 140
        for col in range(imgSrc.shape[1]):
            val = 0
            if vecLineCnt[col] > 0:
                val = vecLineSum[col] // vecLineCnt[col]
                iLastVal = val
            else:
                val = iLastVal
            if val > 255:
                val = 255

            vecLineMean[col] = val

        iLastIdx = int(self.m_param.iKnifeNum) - 1
        iStartPos = self.m_iKnifePos_Cur[0][0]
        iEndPos = self.m_iKnifePos_Cur[iLastIdx][0]
        if iEndPos > iStartPos:
            iTotalSum = sum(vecLineMean[:int(int(iEndPos) - int(iStartPos))])
            self.m_curMembranceMeanGrayValue = iTotalSum // (iEndPos - iStartPos)

        if imgMeanLine is None:
            imgMeanLine = np.empty_like(imgSrc)

        if self.m_icurMeanLineCnt <= 0 or len(self.m_curLineMean) <= 0:
            self.m_curLineMean = vecLineMean
            self.m_icurMeanLineCnt = 1
        else:
            dRate0 = 1.0
            dRate1 = 1.0

            if self.m_icurMeanLineCnt < 100:
                dRate0 = self.m_icurMeanLineCnt / (self.m_icurMeanLineCnt + 1)
                dRate1 = 1.0 / (self.m_icurMeanLineCnt + 1)
                self.m_icurMeanLineCnt += 1
            else:
                if self.m_param.iThresd > 0:
                    dRate0 = 0.65
                    dRate1 = 0.35
                else:
                    dRate0 = 0.5
                    dRate1 = 0.5

            for idx in range(len(self.m_curLineMean)):
                val0 = self.m_curLineMean[idx]
                val1 = vecLineMean[idx]

                val = val0 * dRate0 + val1 * dRate1
                if val < 0:
                    val = 0
                elif val > 255:
                    val = 255
                vecLineMean[idx] = val

        pLineSrc = self.m_curLineMean
        iLen = len(self.m_curLineMean)

        for row in range(imgMeanLine.shape[0]):
            pLinePtr = imgMeanLine[row]
            np.copyto(pLinePtr, pLineSrc, casting='unsafe')

    def _move_imagedata(self, imgSrc):
        iHei = self.m_curImgSize[0] // 2

        rect_dst = (0, 0, self.m_curImgSize[1], iHei)
        rect_src = (0, self.m_curImgSize[0], self.m_curImgSize[1], iHei)
        move_src = self.m_curRecvImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curRecvImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_src = (rect_src[0], rect_src[1] + self.m_curImgSize[0], rect_src[2], rect_src[3])
        rect_dst = (rect_dst[0], rect_dst[1] + self.m_curImgSize[0], rect_dst[2], rect_dst[3])
        move_src = self.m_curRecvImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curRecvImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_dst = (0, iHei, self.m_curImgSize[1], iHei)                            # (0, 1000, 23328, 1000)
        rect_src = (0, self.m_curImgSize[0] + iHei, self.m_curImgSize[1], iHei)     # (0, 3000, 23328, 1000)
        move_src = self.m_curRecvImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curRecvImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_src = (rect_src[0], rect_src[1] + self.m_curImgSize[0], rect_src[2], rect_src[3])  # (0, 5000, 23328, 1000)
        rect_dst = (rect_dst[0], rect_dst[1] + self.m_curImgSize[0], rect_dst[2], rect_dst[3])  # (0, 3000, 23328, 1000)
        move_src = self.m_curRecvImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curRecvImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_dst = (0, 0, self.m_curImgSize[1], iHei)                           # (0,    0, 23328, 1000)
        rect_src = (0, self.m_curImgSize[0], self.m_curImgSize[1], iHei)        # (0, 2000, 23328, 1000)
        move_src = self.m_curBinImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curBinImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_src = (rect_src[0], rect_src[1] + self.m_curImgSize[0], rect_src[2], rect_src[3])  # (0, 4000, 23328, 1000)
        rect_dst = (rect_dst[0], rect_dst[1] + self.m_curImgSize[0], rect_dst[2], rect_dst[3])  # (0, 2000, 23328, 1000)
        move_src = self.m_curBinImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curBinImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_dst = (0, iHei, self.m_curImgSize[1], iHei)                            # (0, 1000, 23328, 1000)
        rect_src = (0, self.m_curImgSize[0] + iHei, self.m_curImgSize[1], iHei)     # (0, 3000, 23328, 1000)
        move_src = self.m_curBinImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curBinImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

        rect_src = (rect_src[0], rect_src[1] + self.m_curImgSize[0], rect_src[2], rect_src[3])
        rect_dst = (rect_dst[0], rect_dst[1] + self.m_curImgSize[0], rect_dst[2], rect_dst[3])
        move_src = self.m_curBinImgBuf[rect_src[1]:rect_src[1]+rect_src[3], rect_src[0]:rect_src[0]+rect_src[2]]
        move_dst = self.m_curBinImgBuf[rect_dst[1]:rect_dst[1]+rect_dst[3], rect_dst[0]:rect_dst[0]+rect_dst[2]]
        np.copyto(move_dst, move_src)

    def _position_segment(self, imgSrc):
        # Assuming CKDImgProc_GetWhiteColEdge is a class and GetPos is a method in that class
        colEdge = CKDImgProc_GetWhiteColEdge()

        bFlag = [False] * 64
        dPos = [[0 for _ in range(2)] for _ in range(64)]
        dLastPos = [[0 for _ in range(2)] for _ in range(64)]
        #
        colEdge.GetPos(bFlag, dPos, imgSrc, self.m_param.iKnifePos, self.m_param.iKnifeNum)

        for idx in range(self.m_param.iKnifeNum):
            if bFlag[idx]:
                self.m_iKnifePos_Cur[idx][0] = dPos[idx][0] + 0.5
                self.m_iKnifePos_Cur[idx][1] = dPos[idx][1] + 0.5
                dLastPos[idx][0] = self.m_iKnifePos_Cur[idx][0]
                dLastPos[idx][1] = self.m_iKnifePos_Cur[idx][1]
            else:
                if dLastPos[idx][0] != 0:
                    self.m_iKnifePos_Cur[idx][0] = dLastPos[idx][0]
                else:
                    self.m_iKnifePos_Cur[idx][0] = (self.m_param.iKnifePos[idx][0] + self.m_param.iKnifePos[idx][
                        1]) * 0.5
                if dLastPos[idx][1] != 0:
                    self.m_iKnifePos_Cur[idx][1] = dLastPos[idx][1]
                else:
                    self.m_iKnifePos_Cur[idx][1] = (self.m_param.iKnifePos[idx][0] + self.m_param.iKnifePos[idx][
                        1]) * 0.5

        if not bFlag[0]:
            self.m_iKnifePos_Cur[0][0] = self.m_param.iKnifePos[0][1]
            self.m_iKnifePos_Cur[0][1] = self.m_param.iKnifePos[0][1]

        iLastIdx = self.m_param.iKnifeNum - 1
        if iLastIdx < 0:
            iLastIdx = 0
        if not bFlag[iLastIdx]:
            self.m_iKnifePos_Cur[iLastIdx][0] = self.m_param.iKnifePos[iLastIdx][0]
            self.m_iKnifePos_Cur[iLastIdx][1] = self.m_param.iKnifePos[iLastIdx][0]

    def _position_segment_firstone(self, pos, iKnifeIdx, imgSrc):
        # 创建ROI区域
        roiRect = (self.m_param.iKnifePos[iKnifeIdx][0], 0, self.m_param.iKnifePos[iKnifeIdx][1] - self.m_param.iKnifePos[iKnifeIdx][0], imgSrc.shape[0])
        roiImg = imgSrc[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]]

        # 初始化直方图
        vecHist = [0] * roiImg.shape[1]

        iLineCnt = 0
        # 计算直方图
        for row in range(0, roiImg.shape[0], 12):
            pLine = roiImg[row]
            for col in range(roiImg.shape[1]):
                vecHist[col] += int(pLine[col])
            iLineCnt += 1

        # 计算直方图的平均值
        drate = 1.0 / iLineCnt
        for col in range(roiImg.shape[1]):
            vecHist[col] *= drate

        # 寻找直方图中大于240的位置
        ret = False
        right = 0
        for col in range(roiImg.shape[1] - 1, -1, -1):
            if vecHist[col] > 240:
                right = col - 1
                ret = True
                break

        # 设置位置
        pos[0] = roiRect[0]
        pos[1] = roiRect[0]
        if ret:
            pos[1] = roiRect[0] + right

        return True

    def _position_segment_one(self, pos, iKnifeIdx, imgSrc):
        # 创建ROI区域
        roiRect = (self.m_param.iKnifePos[iKnifeIdx][0], 0, self.m_param.iKnifePos[iKnifeIdx][1] - self.m_param.iKnifePos[iKnifeIdx][0], imgSrc.shape[0])
        roiImg = imgSrc[roiRect[1]:roiRect[1]+roiRect[3], roiRect[0]:roiRect[0]+roiRect[2]]

        # 初始化直方图
        vecHist = [0] * roiImg.shape[1]

        iLineCnt = 0
        # 计算直方图
        for row in range(0, roiImg.shape[0], 12):
            pLine = roiImg[row]
            for col in range(roiImg.shape[1]):
                vecHist[col] += int(pLine[col])
            iLineCnt += 1

        # 计算直方图的平均值
        drate = 1.0 / iLineCnt
        for col in range(roiImg.shape[1]):
            vecHist[col] *= drate

        # 计算直方图的总和
        iSum = sum(vecHist)

        # 寻找直方图中大于240的位置
        ret = False
        left = 0
        for col in range(roiImg.shape[1]):
            if vecHist[col] > 240:
                left = col
                ret = True
                break

        # 寻找直方图中小于240的位置
        right = left
        for col in range(left + 1, roiImg.shape[1]):
            if vecHist[col] < 240:
                right = col - 1
                break

        # 设置位置
        if ret:
            pos[0] = roiRect[0] + left
            pos[1] = roiRect[0] + right

        return ret


class _CInspDefectData:
    def __init__(self):
        self.iCnt = 0
        self.iRangeX = [100000, 0]  # 0 - left, 1 - right
        self.iRangeY = [100000, 0]  # 0 - top, 1 - bottom

def _InspSeg_UpdateDefectData(data, iX, iY):
    data.iCnt += 1
    if iY < data.iRangeY[0]:
        data.iRangeY[0] = iY
    if iY > data.iRangeY[1]:
        data.iRangeY[1] = iY
    if iX < data.iRangeX[0]:
        data.iRangeX[0] = iX
    if iX > data.iRangeX[1]:
        data.iRangeX[1] = iX

def _InspSeg_CheckDefectDataObj(dfData, param):
    if dfData.iCnt >= param.iMinArea:
        if dfData.iRangeX[1] - dfData.iRangeX[0] + 1 >= param.iMinWid:
            if dfData.iRangeY[1] - dfData.iRangeY[0] + 1 >= param.iMinHei:
                return True
    return False

def _InspSeg_CheckDefectData(iArea, iWid, iHei, param):
    if iArea >= param.iMinArea:
        if iWid >= param.iMinWid:
            if iHei >= param.iMinHei:
                return True
    return False







