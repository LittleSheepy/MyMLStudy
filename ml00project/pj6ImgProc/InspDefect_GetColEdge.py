import cv2
import numpy as np

class CKDImgProc_GetBlackColEdge:
    def __init__(self):
        pass

    def GetPos(self, dOutPos, srcImg_Gray, iRoiX, iPosCount):
        dOutPos.fill(0)
        dPos = np.zeros(iPosCount)
        ret = True
        for idx in range(iPosCount):
            dEdgePos = np.zeros(2)
            if not self._GetEdge_Left2Right(dEdgePos, 0, srcImg_Gray, iRoiX[idx][0], iRoiX[idx][1]):
                ret = False
                break
            if not self._GetEdge_Right2Left(dEdgePos, 1, srcImg_Gray, iRoiX[idx][0], iRoiX[idx][1]):
                ret = False
                break
            dPos[idx] = (dEdgePos[1] + dEdgePos[0]) / 2.0
        if ret:
            np.copyto(dOutPos, dPos)
        return ret

    def _GetEdge_Left2Right(self, pdOutVal, srcImg_Gray, iRoi_StX, iRoi_EndX, iGrad_Thresd=4):
        dRet = iRoi_StX
        tmpImg = srcImg_Gray[iRoi_StX:iRoi_EndX, :]
        roiImg = tmpImg.copy()

        vecColHist, vecGrad = self._GetColHistGrad(roiImg, 0)

        iLen = len(vecGrad) - 1
        iMaxGrad = iGrad_Thresd - 1
        iMaxIdx = -1

        for idx in range(1, iLen):
            if vecGrad[idx] > iMaxGrad:
                iMaxGrad = vecGrad[idx]
                iMaxIdx = idx

        if iMaxIdx <= 0:
            return False

        dot = np.zeros((3, 2), dtype=int)
        dot[0, 0] = iMaxIdx - 1
        dot[0, 1] = vecGrad[iMaxIdx - 1]
        dot[1, 0] = iMaxIdx
        dot[1, 1] = vecGrad[iMaxIdx]
        dot[2, 0] = iMaxIdx + 1
        dot[2, 1] = vecGrad[iMaxIdx + 1]

        pdOutVal[0] = self._GetEdge_SubPixel(dot) + iRoi_StX

        return True

    def _GetEdge_Right2Left(self, pdOutVal, srcImg, iRoi_StX, iRoi_EndX, iGrad_Thresd=4):
        dRet = iRoi_EndX
        tmpImg = srcImg[iRoi_StX:iRoi_EndX, :]
        roiImg = tmpImg.copy()

        vecColHist = []
        vecGrad = []

        self._GetColHistGrad(roiImg, vecColHist, vecGrad, 1)

        iLen = len(vecGrad) - 1
        iMaxGrad = iGrad_Thresd - 1
        iMaxIdx = -1

        for idx in range(iLen, 0, -1):
            if vecGrad[idx] > iMaxGrad:
                iMaxGrad = vecGrad[idx]
                iMaxIdx = idx

        if iMaxIdx <= 0:
            return False

        dot = [0, 0, 0]
        dot[0] = [iMaxIdx - 1, vecGrad[iMaxIdx - 1]]
        dot[1] = [iMaxIdx, vecGrad[iMaxIdx]]
        dot[2] = [iMaxIdx + 1, vecGrad[iMaxIdx + 1]]

        pdOutVal[0] = self._GetEdge_SubPixel(dot) + iRoi_StX

        return True

    def _GetColHistGrad(self, imgRoi, iDir):
        # 对列方向的 作投影累加，求平均
        vecColHist = np.zeros(imgRoi.shape[1], dtype=np.int)
        vecGrad = np.zeros(imgRoi.shape[1], dtype=np.int)

        vecCol = np.sum(imgRoi, axis=0)  # 对列方向累加
        vecCol = vecCol / imgRoi.shape[0]  # 求平均

        vecColHist = vecCol.copy()
        vecColHist[1:-1] = (vecCol[:-2] + vecCol[1:-1] + vecCol[2:]) / 3  # 平滑处理

        if iDir == 0:  # Left 2 Right
            vecGrad[1:-1] = np.maximum(vecColHist[:-2] - vecColHist[2:], 0)
        else:  # Right 2 Left
            vecGrad[1:-1] = np.maximum(vecColHist[2:] - vecColHist[:-2], 0)

        return vecColHist, vecGrad

    def _GetEdge_SubPixel(self, dot):
        if dot[0][1] == dot[1][1] == dot[2][1]:
            return dot[1][0]

        matA = np.array([[dot[i][0]**2, dot[i][0], 1] for i in range(3)], dtype=np.float64)
        matB = np.array([dot[i][1] for i in range(3)], dtype=np.float64).reshape(-1, 1)

        matC = np.linalg.lstsq(matA, matB, rcond=None)[0]  # Solve the system of linear equations

        fMaxPos = -matC[1] / (2 * matC[0])

        return fMaxPos[0]

class CKDImgProc_GetWhiteColEdge:
    def __init__(self):
        pass

    def GetPos(self, bPosFlag, dOutPos, srcImg_Gray, iRoiX, iPosCount):
        ret = True

        for idx in range(iPosCount):
            flag = [False, False]
            dEdgePos = [0, 0]

            ret, dEdgePos[0] = self._GetEdge_Left2Right(srcImg_Gray, iRoiX[idx][0], iRoiX[idx][1])
            if ret:
                flag[0] = True
            ret, dEdgePos[1] = self._GetEdge_Right2Left(srcImg_Gray, iRoiX[idx][0], iRoiX[idx][1])
            if ret:
                flag[1] = True

            if flag[0] and flag[1]:
                dOutPos[idx] = [dEdgePos[0], dEdgePos[1]]
                bPosFlag[idx] = True
            elif flag[0] and not flag[1]:
                dOutPos[idx] = [dEdgePos[0], dEdgePos[0]]
                bPosFlag[idx] = True
            elif not flag[0] and flag[1]:
                dOutPos[idx] = [dEdgePos[1], dEdgePos[1]]
                bPosFlag[idx] = True
            else:
                bPosFlag[idx] = False

        return ret


    def _GetEdge_Left2Right(self, srcImg, iRoi_StX, iRoi_EndX, iGrad_Thresd=20):
        dRet = iRoi_StX

        tmpImg = srcImg[:, iRoi_StX:iRoi_EndX]      # (2000, 100)
        roiImg = tmpImg.copy()

        vecColHist, vecGrad = self._GetColHistGrad(roiImg, 0)

        iLen = len(vecGrad) - 1
        iMaxGrad = iGrad_Thresd - 1
        iMaxIdx = -1

        for idx in range(1, iLen):
            if vecGrad[idx] > iMaxGrad:
                iMaxGrad = vecGrad[idx]
                iMaxIdx = idx

        if iMaxIdx <= 0:
            return False, None

        dot = np.zeros((3,2), dtype=int)
        dot[0, 0] = iMaxIdx - 1
        dot[0, 1] = vecGrad[iMaxIdx - 1]
        dot[1, 0] = iMaxIdx
        dot[1, 1] = vecGrad[iMaxIdx]
        dot[2, 0] = iMaxIdx + 1
        dot[2, 1] = vecGrad[iMaxIdx + 1]

        pdOutVal = self._GetEdge_SubPixel(dot) + iRoi_StX

        return True, pdOutVal

    def _GetEdge_Right2Left(self, srcImg, iRoi_StX, iRoi_EndX, iGrad_Thresd=20):
        dRet = iRoi_EndX

        tmpImg = srcImg[:, iRoi_StX:iRoi_EndX]
        roiImg = tmpImg.copy()

        vecColHist, vecGrad = self._GetColHistGrad(roiImg, 1)

        iLen = len(vecGrad) - 1
        iMaxGrad = iGrad_Thresd - 1
        iMaxIdx = -1

        for idx in range(iLen, 0, -1):
            if vecGrad[idx] > iMaxGrad:
                iMaxGrad = vecGrad[idx]
                iMaxIdx = idx

        if iMaxIdx <= 0:
            return False, None

        dot = np.zeros((3,2), dtype=int)
        dot[0, 0] = iMaxIdx - 1
        dot[0, 1] = vecGrad[iMaxIdx - 1]
        dot[1, 0] = iMaxIdx
        dot[1, 1] = vecGrad[iMaxIdx]
        dot[2, 0] = iMaxIdx + 1
        dot[2, 1] = vecGrad[iMaxIdx + 1]

        pdOutVal = self._GetEdge_SubPixel(dot) + iRoi_StX

        return True, pdOutVal


    def _GetColHistGrad(self, imgRoi, iDir):
        vecColHist = np.zeros(imgRoi.shape[1], dtype=int)
        vecGrad = np.zeros(imgRoi.shape[1], dtype=int)

        vecCol = np.zeros(imgRoi.shape[1], dtype=int)

        for row in range(0, imgRoi.shape[0], 8):
            pLine = imgRoi[row, :]
            for col in range(imgRoi.shape[1]):
                vecCol[col] += pLine[col]

        dRate = 1.0 / (imgRoi.shape[0] >> 3)
        vecCol = vecCol * dRate

        vecColHist = vecCol.copy()
        iSubLen = imgRoi.shape[1] - 1
        for col in range(1, iSubLen):
            vecColHist[col] = (vecCol[col - 1] + vecCol[col] + vecCol[col + 1]) / 3
        diff = 0
        iLen = len(vecColHist) - 1
        if iDir == 0:  # Left 2 Right
            for idx in range(1, iLen):
                diff = vecColHist[idx + 1] - vecColHist[idx - 1]
                vecGrad[idx] = max(diff, 0)
        else:  # Right 2 Left
            for idx in range(1, iLen):
                diff = vecColHist[idx - 1] - vecColHist[idx + 1]
                vecGrad[idx] = max(diff, 0)

        return vecColHist, vecGrad

    def _GetEdge_SubPixel(self, dot):
        if (dot[0, 1] == dot[1, 1]) and (dot[1, 1] == dot[2, 1]):
            return dot[1, 0]

        matA = np.zeros((3, 3))         # [[0. 0. 0.], [0. 0. 0.], [0. 0. 0.]]
        matB = np.zeros((3, 1))         # [[0.], [0.], [0.]]

        for idx in range(3):
            matA[idx, 0] = dot[idx, 0] ** 2
            matA[idx, 1] = dot[idx, 0]
            matA[idx, 2] = 1

            matB[idx, 0] = dot[idx, 1]

        fMaxPos = dot[1, 0]

        # y = ax2 + bx + c
        matC, _, _, _ = np.linalg.lstsq(matA, matB, rcond=None)
        da = matC[0, 0]
        db = matC[1, 0]

        fMaxPos = -db * 0.5 / da

        return fMaxPos