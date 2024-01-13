import cv2
import numpy as np
from typing import List
class AM_TLine:
    y = 0
    xSt = 0
    xEnd = 0

class AM_TPoint:
    x = 0
    y = 0

class CAM_BlobSet:
    def __init__(self):
        self.m_listElement = []

    def clear(self):
        self.m_listElement.clear()

    def get_total(self):
        return len(self.m_listElement)

    def add(self, blob):
        self.m_listElement.append(blob)
        return len(self.m_listElement) - 1

    def add_with_offset(self, pBlob, iOffs_x, iOffs_y):
        blob = pBlob
        blob.Offs(iOffs_x, iOffs_y)
        self.m_listElement.append(blob)
        return len(self.m_listElement) - 1

    def add_blob_set(self, blobSet):
        self.m_listElement.extend(blobSet.m_listElement)
        return len(self.m_listElement) - 1

    def get_element(self, iIndex):
        assert iIndex < len(self.m_listElement)
        return self.m_listElement[iIndex]

    def get_match_blob_idx(self, cent):
        rtn = -1
        for idx in range(len(self.m_listElement)):
            if self.m_listElement[idx].IsInRect(cent.x, cent.y):
                rtn = idx
                break
        return rtn

    def get_rect_blob_idx(self, listBlobIdx, rect):
        listBlobIdx.clear()
        iLen = len(self.m_listElement)
        for idx in range(iLen):
            cent = self.m_listElement[idx].GetCent()
            if rect.IsPtInRect(cent.x, cent.y):
                listBlobIdx.append(idx)

    def input_merge_blob(self, pBlob, iMergeDist):
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
    line = AM_TLine()
    wid = 0
    iCurObjIdx = 0
    iNewObjIdx = 0

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
        blob_set.clear()

        if bin_img.shape[0] <= 0 or bin_img.shape[1] <= 0:
            return False

        match_obj_vec = []
        last_match_line = []
        new_match_line = []

        for row in range(4000, bin_img.shape[0]): # 6000
            p_line = bin_img[row, :]    # 2571

            # Implement your _ScanRow, _MatchLine, _UpdateMerge, _UpdateObjVec, _UpdateMatchLine logic here
            CAMALG_GetBlobList._ScanRow(new_match_line, p_line, bin_img.shape[1], row)

            i_len = len(new_match_line)
            i_last_st_idx = 0

            for idx in range(i_len):
                i_last_st_idx = CAMALG_GetBlobList._MatchLine(i_last_st_idx, last_match_line, new_match_line, idx)

            CAMALG_GetBlobList._UpdateMerge(match_obj_vec, last_match_line)  # Merge two objs
            CAMALG_GetBlobList._UpdateObjVec(match_obj_vec, new_match_line)  # newLine -> input the obj
            CAMALG_GetBlobList._UpdateMatchLine(new_match_line, last_match_line)
            # Implement your _ScanRow, _MatchLine, _UpdateMerge, _UpdateObjVec, _UpdateMatchLine logic here

        blob_set.clear()

        for idx in range(len(match_obj_vec)):
            if match_obj_vec[idx].m_bBlobValid:
                match_obj_vec[idx].calc()

                if match_obj_vec[idx].get_area() >= i_min_area:
                    blob_set.add(match_obj_vec[idx])

        match_obj_vec.clear()
        last_match_line.clear()
        new_match_line.clear()

        return True, blob_set

    @staticmethod
    def _ScanRow(lineVec, pLine, iWid, y):
        lineVec.clear()

        state = 0

        line = _AM_TMatchLine()

        iXSt = 0
        iXEnd = iWid // 4

        for idx in range(iXSt, iXEnd):
            val = pLine[idx]

            if val != 0:
                iOff_x = idx * 4
                print(iOff_x)
                pTmp = pLine[iOff_x:iOff_x+4]
                for loop in range(4):
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
                    # try:
                    if iOff_x is None:
                        iOff_x = 0
                    iOff_x += 1
                    # except Exception:
                    #     print("")

        if state == 1:
            line.y = y
            state = 0

            lineVec.append(line)

        return lineVec

    @staticmethod
    def _MatchDir(last_line, new_line):
        if new_line.line.x_end + 1 < last_line.line.x_st:
            return 1
        elif new_line.line.x_st > last_line.line.x_end + 1:
            return -1
        return 0

    @staticmethod
    def _UpdateLastLineObjIdx(last_vec, src_obj_idx, dst_obj_idx, st_idx, end_idx):
        for idx in range(st_idx, end_idx + 1):
            if src_obj_idx == last_vec[idx].new_obj_idx:
                last_vec[idx].new_obj_idx = dst_obj_idx

    @staticmethod
    def _UpdateNewLineObjIdx(new_vec, src_obj_idx, dst_obj_idx, st_idx, end_idx):
        for idx in range(st_idx, end_idx + 1):
            if src_obj_idx == new_vec[idx].cur_obj_idx:
                new_vec[idx].cur_obj_idx = dst_obj_idx

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

                        _update_last_line_obj_idx(lastVec, maxIdx, minIdx, 0, len_lastVec - 1)  # [ 0, idx ]
                        _update_new_line_obj_idx(newVec, maxIdx, minIdx, 0, iNew_CurIdx)  # [ 0, iNew_CurIdx ]
            elif ret > 0:
                break
            else:
                iCurMatchIdx = idx

        piLastStIdx[0] = iCurMatchIdx

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
            else:
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

