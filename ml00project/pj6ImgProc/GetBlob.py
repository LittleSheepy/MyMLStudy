import cv2
import numpy as np
from typing import List
class AM_TLine:
    def __init__(self):
        self.y = 0
        self.xSt = 0
        self.xEnd = 0

class AM_TPoint:
    x = 0
    y = 0

class AM_TRect:
    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def IsPtInRect(self, ix, iy):
        return (ix >= self.x) and (ix < self.x + self.width) and (iy >= self.y) and (iy < self.y + self.height)

    def GetDistance(self, rect):
        dist_x = 0
        if rect.x > self.x + self.width:
            dist_x = rect.x - self.x - self.width
        elif self.x > rect.x + rect.width:
            dist_x = self.x - rect.x - rect.width

        dist_y = 0
        if rect.y > self.y + self.height:
            dist_y = rect.y - self.y - self.height
        elif self.y > rect.y + rect.height:
            dist_y = self.y - rect.y - rect.height

        return dist_x + dist_y

class CAM_Blob:
    def __init__(self):
        self.m_iArea = 0
        self.m_Rect = AM_TRect()
        self.m_Cent = AM_TPoint()
        self.m_bBlobValid = False
        self.m_iNeibBlobCnt = 0
        self.m_bIsLikeLine = False
        self.m_listElement = []

    def GetArea(self):
        return self.m_iArea

    def GetAreaWithImg(self, validPxImg):
        total = self.GetTotal()
        area = 0
        for idx in range(total):
            pLine = self.GetElement(idx)
            xEnd = pLine.xEnd + 1
            pValidImgLine = validPxImg[pLine.y]
            for x in range(pLine.xSt, xEnd):
                if pValidImgLine[x] > 0:
                    area += 1
        return area

    def GetRect(self):
        return self.m_Rect

    def GetCent(self):
        return self.m_Cent

    def Offs(self, iOffsX, iOffsY):
        self.m_Rect.x += iOffsX
        self.m_Rect.y += iOffsY
        self.m_Cent.x += iOffsX
        self.m_Cent.y += iOffsY
        for idx in range(len(self.m_listElement)):
            self.m_listElement[idx].xSt += iOffsX
            self.m_listElement[idx].xEnd += iOffsX
            self.m_listElement[idx].y += iOffsY

    def GetTotal(self):
        return len(self.m_listElement)

    def Clear(self):
        self.m_iArea = 0
        self.m_listElement.clear()

    def GetElement(self, iIndex):
        assert iIndex < len(self.m_listElement)
        return self.m_listElement[iIndex]

    def AddElement(self, line):
        self.m_listElement.append(line)

    def MergeBlob(self, blob):
        self.m_listElement.extend(blob.m_listElement)
        self.m_iArea += blob.GetArea()
        minx = min(self.m_Rect.x, blob.m_Rect.x)
        maxx = max(self.m_Rect.x+self.m_Rect.width, blob.m_Rect.x + blob.m_Rect.width)
        miny = min(self.m_Rect.y, blob.m_Rect.y)
        maxy = max(self.m_Rect.y + self.m_Rect.height, blob.m_Rect.y + blob.m_Rect.height)
        self.m_Rect.x = minx
        self.m_Rect.y = miny
        self.m_Rect.width = maxx - minx
        self.m_Rect.height = maxy - miny
        self.m_Cent.x = self.m_Rect.x + (self.m_Rect.width // 2)
        self.m_Cent.y = self.m_Rect.y + (self.m_Rect.height // 2)

    def IsInRect(self, ix, iy):
        return (ix >= self.m_Rect.x) and (ix < self.m_Rect.x + self.m_Rect.width) and (iy >= self.m_Rect.y) and (iy < self.m_Rect.y + self.m_Rect.height)

    def Calc(self):
        sum = 0
        len = self.GetTotal()
        if len <= 0:
            self.m_iArea = 0
            self.m_Rect.width = 0
            self.m_Rect.height = 0
            return
        x_min = self.m_listElement[0].xSt
        x_max = self.m_listElement[0].xEnd
        y_min = self.m_listElement[0].y
        y_max = self.m_listElement[len - 1].y
        for idx in range(len):
            sum += self.m_listElement[idx].xEnd - self.m_listElement[idx].xSt + 1
            if self.m_listElement[idx].xSt < x_min:
                x_min = self.m_listElement[idx].xSt
            if self.m_listElement[idx].xEnd > x_max:
                x_max = self.m_listElement[idx].xEnd
        self.m_iArea = sum
        self.m_Rect.x = x_min
        self.m_Rect.y = y_min
        self.m_Rect.width = x_max - x_min + 1
        self.m_Rect.height = y_max - y_min + 1
        self.m_Cent.x = self.m_Rect.x + (self.m_Rect.width // 2)
        self.m_Cent.y = self.m_Rect.y + (self.m_Rect.height // 2)

    def CheckLikeLine(self):
        if self.m_iArea < 600:
            self.m_bIsLikeLine = False
            return

        vecDot = []
        total = self.GetTotal()
        step = 1
        if total > 256:
            step = 4
        for idx in range(total):
            pLine = self.GetElement(idx)
            fY = pLine.y
            for col in range(pLine.xSt, pLine.xEnd + 1, step):
                vecDot.append((col, fY))

        line = cv2.fitLine(np.array(vecDot, dtype=np.float32), cv2.DIST_L2, 0, 0.01, 0.01)

        if abs(line[0]) > abs(line[1]):
            self.m_bIsLikeLine = False
            return

        total = len(vecDot)
        cnt = 0
        for idx in range(total):
            dist = self._AMBlob_getDist_Dot2Line(line, vecDot[idx])
            if dist < 50:
                cnt += 1

        if cnt > total * 0.6:
            self.m_bIsLikeLine = True

    def draw(self, binImg, val):
        iLineCount = len(self.m_listElement)
        for col in range(iLineCount):
            line = self.m_listElement[col]
            cv2.line(binImg, (line.xSt, line.y), (line.xEnd, line.y), val)

class CAM_BlobSet:
    def __init__(self):
        self.m_listElement = []

    def Clear(self):
        self.m_listElement.clear()

    def GetTotal(self):
        return len(self.m_listElement)

    def Add(self, blob):
        self.m_listElement.append(blob)
        return len(self.m_listElement) - 1

    def Add_with_offset(self, pBlob, iOffs_x, iOffs_y):
        blob = pBlob
        blob.Offs(iOffs_x, iOffs_y)
        self.m_listElement.append(blob)
        return len(self.m_listElement) - 1

    def Add_blob_set(self, blobSet):
        self.m_listElement.extend(blobSet.m_listElement)
        return len(self.m_listElement) - 1

    def GetElement(self, iIndex):
        assert iIndex < len(self.m_listElement)
        return self.m_listElement[iIndex]

    def GetMatchBlobIdx(self, cent):
        rtn = -1
        for idx in range(len(self.m_listElement)):
            if self.m_listElement[idx].IsInRect(cent.x, cent.y):
                rtn = idx
                break
        return rtn

    def GetRectBlobIdx(self, listBlobIdx, rect):
        listBlobIdx.clear()
        iLen = len(self.m_listElement)
        for idx in range(iLen):
            cent = self.m_listElement[idx].GetCent()
            if rect.IsPtInRect(cent.x, cent.y):
                listBlobIdx.append(idx)

    def InputMergeBlob(self, pBlob, iMergeDist):
        rect_I = pBlob.GetRect()
        flag = False
        total = len(self.m_listElement)
        for idx in range(total):
            rt = self.m_listElement[idx].GetRect()
            if rect_I.GetDistance(rt) < iMergeDist:
                self.m_listElement[idx].MergeBlob(pBlob)
                self.m_listElement[idx].m_iNeibBlobCnt += 1
                flag = True
                break
        if not flag:
            self.m_listElement.append(pBlob)

class _AM_TMatchLine:
    def __init__(self):
        self.line = AM_TLine()
        self.wid = 0
        self.iCurObjIdx = 0
        self.iNewObjIdx = 0

def ScanRow(lineVec, pLine, iWid, y):
    # print("> <func> _ScanRow")
    lineVec.clear()

    state = 0

    line = _AM_TMatchLine()

    iXSt = 0
    iXEnd = iWid // 4
    iOff_x = 0
    # print("======== iXEnd= ", iXEnd)
    for idx in range(iXSt, iXEnd):
        iOff_x = idx * 4
        pTmp = pLine[iOff_x:iOff_x+4]
        # print("_ScanRow idx=", idx)
        val = pTmp.sum()
        if idx == iXEnd-1:
            pass
        if val != 0:
            for loop in range(4):
                # print("       _ScanRow loop=", loop)
                iOff_x = idx * 4 + loop
                if state == 0:
                    if pTmp[loop] != 0:
                        line.line.xSt = iOff_x
                        line.line.xEnd = iOff_x
                        state = 1
                elif state == 1:
                    if pTmp[loop] == 0:
                        state = 0

                        line.line.y = y

                        line.iCurObjIdx = -1
                        line.iNewObjIdx = -1
                        lineVec.append(line)
                        line = _AM_TMatchLine()
                        # line = Line(0, 0, y)
                    else:
                        if iOff_x - line.line.xEnd <= 1:
                            line.line.xEnd = iOff_x
                        else:
                            line.line.y = y
                            line.iCurObjIdx = -1
                            line.iNewObjIdx = -1

                            lineVec.append(line)
                            line = _AM_TMatchLine()

                            line.line.xSt = iOff_x
                            line.line.xEnd = iOff_x

                # iOff_x += 1

    # print("======== for end= ")
    if state == 1:
        line.line.y = y
        state = 0
        line.iCurObjIdx = -1
        line.iNewObjIdx = -1

        lineVec.append(line)
        line = _AM_TMatchLine()
    # print("< <func> _ScanRow")
    return lineVec

class CAMALG_GetBlobList:
    def __init__(self):
        pass

    @staticmethod
    def GetBlobList_BinaryImage(blob_set, bin_img, i_min_area):
        blob_set.clear()

        if bin_img.shape[0] <= 0 or bin_img.shape[1] <= 0:
            return False

        match_obj_vec = []
        last_match_line = []
        new_match_line = []

        for row in range(bin_img.shape[0]):
            p_line = bin_img[row, :]

            # Implement your _ScanRow, _MatchLine, _UpdateMerge, _UpdateObjVec, _UpdateMatchLine logic here

        blob_set.clear()

        for idx in range(len(match_obj_vec)):
            if match_obj_vec[idx].m_bBlobValid:
                match_obj_vec[idx].calc()

                if match_obj_vec[idx].get_area() > i_min_area:
                    rt = match_obj_vec[idx].get_rect()

                    drate = rt[2] / rt[3]
                    if 0.5 < drate < 2.0:
                        blob_set.add(match_obj_vec[idx])

        match_obj_vec.clear()
        last_match_line.clear()
        new_match_line.clear()

        return True

    @staticmethod
    def GetBlobList_BinaryImage_forAll(blob_set, bin_img, i_min_area):
        print("> <func> GetBlobList_BinaryImage_forAll")
        blob_set.Clear()

        if bin_img.shape[0] <= 0 or bin_img.shape[1] <= 0:
            return False

        match_obj_vec = []
        last_match_line = []
        new_match_line = []

        for row in range(0, bin_img.shape[0]): # 6000
            p_line = bin_img[row, :]    # 2571

            # Implement your _ScanRow, _MatchLine, _UpdateMerge, _UpdateObjVec, _UpdateMatchLine logic here
            ScanRow(new_match_line, p_line, bin_img.shape[1], row)

            i_len = len(new_match_line)
            i_last_st_idx = 0

            for idx in range(i_len):
                i_last_st_idx = CAMALG_GetBlobList._MatchLine(i_last_st_idx, last_match_line, new_match_line, idx)

            CAMALG_GetBlobList._UpdateMerge(match_obj_vec, last_match_line)  # Merge two objs 合并二个obj
            CAMALG_GetBlobList._UpdateObjVec(match_obj_vec, new_match_line)  # newLine -> input the obj  matchObjVec
            CAMALG_GetBlobList._UpdateMatchLine(new_match_line, last_match_line)
            # Implement your _ScanRow, _MatchLine, _UpdateMerge, _UpdateObjVec, _UpdateMatchLine logic here

        blob_set.Clear()

        for idx in range(len(match_obj_vec)):
            if match_obj_vec[idx].m_bBlobValid:
                match_obj_vec[idx].Calc()

                if match_obj_vec[idx].GetArea() >= i_min_area:  # >=3
                    blob_set.Add(match_obj_vec[idx])

        match_obj_vec.clear()
        last_match_line.clear()
        new_match_line.clear()

        print("< <func> GetBlobList_BinaryImage_forAll")
        return True

    @staticmethod
    def ScanRow(lineVec, pLine, iWid, y):
        print("> <func> _ScanRow")
        lineVec.clear()

        state = 0

        line = _AM_TMatchLine()

        iXSt = 0
        iXEnd = iWid // 4
        iOff_x = 0
        print("======== iXEnd= ", iXEnd)
        for idx in range(iXSt, iXEnd):
            iOff_x = idx * 4
            pTmp = pLine[iOff_x:iOff_x+4]
            # print("_ScanRow idx=", idx)
            val = pTmp.sum()
            if idx == iXEnd-1:
                pass
            if val != 0:
                for loop in range(4):
                    # print("       _ScanRow loop=", loop)
                    iOff_x = idx * 4 + loop
                    if state == 0:
                        if pTmp[loop] != 0:
                            line.line.xSt = iOff_x
                            line.line.xEnd = iOff_x
                            state = 1
                    elif state == 1:
                        if pTmp[loop] == 0:
                            state = 0

                            line.line.y = y

                            line.iCurObjIdx = -1
                            line.iNewObjIdx = -1
                            lineVec.append(line)
                            # line = Line(0, 0, y)
                        else:
                            if iOff_x - line.line.xEnd <= 1:
                                line.line.xEnd = iOff_x
                            else:
                                line.line.y = y
                                line.iCurObjIdx = -1
                                line.iNewObjIdx = -1

                                lineVec.append(line)

                                line.line.xSt = iOff_x
                                line.line.xEnd = iOff_x

                    # iOff_x += 1

        print("======== for end= ")
        if state == 1:
            line.line.y = y
            state = 0
            line.iCurObjIdx = -1
            line.iNewObjIdx = -1

            lineVec.append(line)
        print("< <func> _ScanRow")
        return lineVec

    @staticmethod
    def _MatchDir(last_line, new_line):
        if new_line.line.xEnd + 1 < last_line.line.xSt:
            return 1
        elif new_line.line.xSt > last_line.line.xEnd + 1:
            return -1
        return 0

    @staticmethod
    def _UpdateLastLineObjIdx(last_vec, src_obj_idx, dst_obj_idx, st_idx, end_idx):
        for idx in range(st_idx, end_idx + 1):
            if src_obj_idx == last_vec[idx].iNewObjIdx:
                last_vec[idx].iNewObjIdx = dst_obj_idx

    @staticmethod
    def _UpdateNewLineObjIdx(new_vec, src_obj_idx, dst_obj_idx, st_idx, end_idx):
        for idx in range(st_idx, end_idx + 1):
            if src_obj_idx == new_vec[idx].iCurObjIdx:
                new_vec[idx].iCurObjIdx = dst_obj_idx

    @staticmethod
    def _MatchLine(piLastStIdx, lastVec, newVec, iNew_CurIdx):
        ret = 0
        len_lastVec = len(lastVec)
        iStIdx = piLastStIdx
        iCurMatchIdx = 0

        for idx in range(iStIdx, len_lastVec):
            ret = CAMALG_GetBlobList._MatchDir(lastVec[idx], newVec[iNew_CurIdx])

            if ret == 0:
                iCurMatchIdx = idx

                if newVec[iNew_CurIdx].iCurObjIdx == -1:
                    # 未标记
                    newVec[iNew_CurIdx].iCurObjIdx = lastVec[idx].iNewObjIdx
                else:
                    if newVec[iNew_CurIdx].iCurObjIdx != lastVec[idx].iNewObjIdx:
                        # 不等，且已有标记
                        minIdx = min(newVec[iNew_CurIdx].iCurObjIdx, lastVec[idx].iNewObjIdx)
                        maxIdx = max(newVec[iNew_CurIdx].iCurObjIdx, lastVec[idx].iNewObjIdx)

                        CAMALG_GetBlobList._UpdateLastLineObjIdx(lastVec, maxIdx, minIdx, 0, len_lastVec - 1)  # [ 0, idx ]
                        CAMALG_GetBlobList._UpdateNewLineObjIdx(newVec, maxIdx, minIdx, 0, iNew_CurIdx)  # [ 0, iNew_CurIdx ]
            elif ret > 0:
                break
            else:
                iCurMatchIdx = idx

        piLastStIdx = iCurMatchIdx
        return piLastStIdx

    @staticmethod
    def _MergeObj(dstObj, curObj):
        if curObj.m_bBlobValid:
            dstObj.MergeBlob(curObj)
            curObj.Clear()
            curObj.m_bBlobValid = False

    @staticmethod
    def _IsMatch(line0, line1):
        sumLen = line0.wid + line1.wid
        min_x = min(line0.line.xSt, line1.line.xSt)
        max_x = max(line0.line.xEnd, line1.line.xEnd)
        ret = True
        if sumLen <= (max_x - min_x + 1):
            ret = False
        return ret

    @staticmethod
    def _UpdateMerge(matchObjVec, lastMatchLine):
        ilen = len(lastMatchLine)
        for idx in range(ilen):
            if lastMatchLine[idx].iCurObjIdx != lastMatchLine[idx].iNewObjIdx:
                CAMALG_GetBlobList._MergeObj(matchObjVec[lastMatchLine[idx].iNewObjIdx],
                                             matchObjVec[lastMatchLine[idx].iCurObjIdx])

    @staticmethod
    def _UpdateObjVec(matchObjVec, newMatchLine):
        iLen = len(newMatchLine)
        for idx in range(iLen):
            if newMatchLine[idx].iCurObjIdx != -1:
                assert matchObjVec[newMatchLine[idx].iCurObjIdx].m_bBlobValid
                matchObjVec[newMatchLine[idx].iCurObjIdx].AddElement(newMatchLine[idx].line)
            else: # 未匹配 分配对象编号
                obj = CAM_Blob()
                matchObjVec.append(obj)
                idx_t = len(matchObjVec) - 1
                matchObjVec[idx_t].m_bBlobValid = True
                iObjIdx = len(matchObjVec) - 1
                matchObjVec[iObjIdx].AddElement(newMatchLine[idx].line)
                newMatchLine[idx].iCurObjIdx = iObjIdx

    @staticmethod
    def _UpdateMatchLine(newMatchLine, lastMatchLine):
        lastMatchLine.clear()
        lastMatchLine.extend(newMatchLine)
        ilen = len(lastMatchLine)
        for idx in range(ilen):
            lastMatchLine[idx].iNewObjIdx = lastMatchLine[idx].iCurObjIdx

    @staticmethod
    def DrawBlob(dst_img, blob_set):
        i_blob_count = blob_set.get_total()

        for idx in range(i_blob_count):
            blob = blob_set.get_element(idx)

            i_line_count = blob.get_total()
            for col in range(i_line_count):
                line = blob.get_element(col)
                cv2.line(dst_img, (line.xSt, line.y), (line.xEnd, line.y), (255, 255, 255))

    # ... rest of the code ...

