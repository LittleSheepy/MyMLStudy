import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import random as rng

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

cnt = 0
def pltShowCVDot(img, point, title="img"):
    global cnt
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        print('Image is single-channel (grayscale)')
    elif len(img.shape) == 3:
        print('Image is three-channel (RGB)')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.circle(img, list(point), 20, (255, 0, 0), -1)
    plt.imshow(img)
    plt.title(title + str(cnt))
    plt.show()
    cnt += 1

def draw_contours(img_gray, contours, hierarchy):
    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
    return drawing

class _TKDWater_Defect2:
    def __init__(self):
        self.rtRect = None
        self.iArea = 0
        self.iPos = 0
        self.bNIC = False
        self.ptCenter = None

class _CKDWater_MatchBox:
    def __init__(self):
        self.m_iMeanVal = 80
        self.m_iRow = 0
        self.m_iCol = 0
        self.ptCent = [0, 0]
        self.imgMatch = None
        self.imgBoxFull = None
        self.m_vptNIC = []
        self.m_vptNIC.append([124, 124])

        self.m_vptNIC.append([65, 100])
        self.m_vptNIC.append([65, 148])
        self.m_vptNIC.append([100, 183])
        self.m_vptNIC.append([148, 183])

        self.m_vptNIC.append([183, 148])
        self.m_vptNIC.append([183, 100])
        self.m_vptNIC.append([148, 65])
        self.m_vptNIC.append([100, 65])
        self.m_nic_mean = []

    def IsPtInRect(self, pt):
        ret = False
        iHalfW = self.imgMatch.shape[1] >> 1
        iHalfH = self.imgMatch.shape[0] >> 1

        if (pt[0] >= self.ptCent[0] - iHalfW and
            pt[1] >= self.ptCent[1] - iHalfH and
            pt[0] < self.ptCent[0] + iHalfW and
            pt[1] < self.ptCent[1] + iHalfH):
            ret = True

        return ret

    def CalcMeanVal(self):
        global iDir_g
        _, imgBin = cv2.threshold(self.imgMatch, 0, 255, cv2.THRESH_OTSU)

        val = cv2.mean(self.imgMatch, imgBin)[0]
        if val < 30.0:
            val = 30

        self.m_iMeanVal = int(val)

        rate = 100.0 / val

        self.imgMatch = self.imgMatch.astype('float64')
        self.imgMatch *= rate
        self.imgMatch = self.imgMatch.astype('uint8')
        #pltShowCV(self.imgMatch)

        for i in range(9):
            nic_center_pt = self.m_vptNIC[i]

            zero = 1.85
            radio = 2.42
            if iDir_g == 0:
                nic_offset_x = int((zero - self.m_iCol)*radio)
                nic_offset_y = int((zero - self.m_iRow)*radio)
            elif iDir_g == 1:
                nic_offset_x = int((zero - self.m_iCol)*radio)
                nic_offset_y = int((self.m_iRow-zero)*radio)
            elif iDir_g == 2:
                nic_offset_x = int((self.m_iCol - zero)*radio)
                nic_offset_y = int((zero - self.m_iRow)*radio)
            else:
                nic_offset_x = int((self.m_iCol - zero)*radio)
                nic_offset_y = int((self.m_iRow-zero)*radio)
            # nic_offset_x = int((3 - self.m_iCol)*1.7)
            # nic_offset_y = int((3 - self.m_iRow)*1.7)
            # nic_offset_x = 0
            # nic_offset_y = 0
            nic_center_pt = [nic_center_pt[0] + nic_offset_x, nic_center_pt[1] + nic_offset_y]
            mean = calculate_mean(self.imgMatch, nic_center_pt)
            self.m_nic_mean.append(mean)


def calculate_mean(gray_image, center, diameter=36):
    radius = diameter // 2
    mask = np.zeros(gray_image.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    masked_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    mean_value = cv2.mean(masked_image, mask=mask)[0]
    return mean_value

class CAlgo_KDWater2:
    def __init__(self):
        self.m_imgTempl = None
        self.m_imgTempl_tabMask_Col = None
        self.m_imgTempl_tabMask_Row = None
        self.m_ptStdOffs = None
        self.vecBox = None
        self.m_vptNIC = []
        self.m_vptNIC.append([124, 124])

        self.m_vptNIC.append([65, 100])
        self.m_vptNIC.append([65, 148])
        self.m_vptNIC.append([100, 183])
        self.m_vptNIC.append([148, 183])

        self.m_vptNIC.append([183, 148])
        self.m_vptNIC.append([183, 100])
        self.m_vptNIC.append([148, 65])
        self.m_vptNIC.append([100, 65])

    def LoadTemplate(self):
        self.m_imgTempl = cv2.imread("Template/nm_block.bmp", cv2.IMREAD_GRAYSCALE)
        if self.m_imgTempl is None:
            return False

        self.m_ptStdOffs = (213, 213)
        self.m_imgTempl_tabMask_Col = cv2.imread("Template/nm_tabmask_col.bmp", cv2.IMREAD_GRAYSCALE)
        self.m_imgTempl_tabMask_Row = cv2.imread("Template/nm_tabmask_row.bmp", cv2.IMREAD_GRAYSCALE)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.m_imgTempl_tabMask_Col = cv2.erode(self.m_imgTempl_tabMask_Col, kernel)
        self.m_imgTempl_tabMask_Row = cv2.erode(self.m_imgTempl_tabMask_Row, kernel)

        if (self.m_imgTempl_tabMask_Row is None or
            self.m_imgTempl_tabMask_Row.shape != self.m_imgTempl.shape):
            self.m_imgTempl_tabMask_Row = np.zeros(self.m_imgTempl.shape, np.uint8)
            self.m_imgTempl_tabMask_Row.fill(255)

        if (self.m_imgTempl_tabMask_Col is None or
            self.m_imgTempl_tabMask_Col.shape != self.m_imgTempl.shape):
            self.m_imgTempl_tabMask_Col = np.zeros(self.m_imgTempl.shape, np.uint8)
            self.m_imgTempl_tabMask_Col.fill(255)

        return True

    def FindDefect(self, vecDefect, srcImgGray, i_dir):
        vecDefect.clear()
        if self.m_imgTempl is None or self.m_imgTempl_tabMask_Col is None or self.m_imgTempl_tabMask_Row is None:
            return False, vecDefect

        ptCent = [0, 0]
        rtCent = (
            self.m_imgTempl.shape[1],
            self.m_imgTempl.shape[0],
            srcImgGray.shape[1] - self.m_imgTempl.shape[1]*2,
            srcImgGray.shape[0] - self.m_imgTempl.shape[0]*2)
        srcImgGray = cv2.GaussianBlur(srcImgGray, (7, 7), 5.0)

        if self._GetBestMatchPos(ptCent, srcImgGray[rtCent[1]:rtCent[1] + rtCent[3], rtCent[0]:rtCent[0] + rtCent[2]]):
            box = _CKDWater_MatchBox()
            rtCent_tl = [rtCent[0], rtCent[1]]
            box.ptCent = np.array(ptCent) + np.array(rtCent_tl)

            curRect = (
                box.ptCent[0] - (self.m_imgTempl.shape[1] >> 1),
                box.ptCent[1] - (self.m_imgTempl.shape[0] >> 1),
                self.m_imgTempl.shape[1],
                self.m_imgTempl.shape[0])
            box.imgMatch = srcImgGray[curRect[1]:curRect[1] + curRect[3], curRect[0]:curRect[0] + curRect[2]]
            pltShowCVDot(srcImgGray, box.ptCent, "srcImgGray")
            box.m_iRow = 0
            box.m_iCol = 0
            vecDefect = self._SearchDefect(vecDefect, box, srcImgGray, i_dir)

        if len(vecDefect) == 0:
            return True, vecDefect

        # Post-processing
        if len(vecDefect) == 1:
            if vecDefect[0].iPos == 2:
                return True, vecDefect
            else:
                return False, vecDefect
        elif len(vecDefect) == 2:
            if (vecDefect[0].iPos == 2) or (vecDefect[1].iPos == 2):
                return True, vecDefect
            else:
                return False, vecDefect
        else:
            return False, vecDefect
        return False, vecDefect

    def _UniteSortRowCol(self, vec_box, i_dir):
        i_total = len(vec_box)
        if i_total <= 0:
            return

        i_min_x = vec_box[0].m_iCol
        i_max_x = vec_box[0].m_iCol
        i_min_y = vec_box[0].m_iRow
        i_max_y = vec_box[0].m_iRow

        for idx in range(1, i_total):
            i_cur_x = vec_box[idx].m_iCol
            i_cur_y = vec_box[idx].m_iRow

            if i_cur_x < i_min_x:
                i_min_x = i_cur_x
            if i_cur_x > i_max_x:
                i_max_x = i_cur_x
            if i_cur_y < i_min_y:
                i_min_y = i_cur_y
            if i_cur_y > i_max_y:
                i_max_y = i_cur_y

        if i_dir == 0:  # LT
            for idx in range(i_total):
                vec_box[idx].m_iCol = vec_box[idx].m_iCol - i_min_x
                vec_box[idx].m_iRow = vec_box[idx].m_iRow - i_min_y
        elif i_dir == 1:  # LB
            for idx in range(i_total):
                vec_box[idx].m_iCol = vec_box[idx].m_iCol - i_min_x
                vec_box[idx].m_iRow = i_max_y - vec_box[idx].m_iRow
        elif i_dir == 2:  # RT
            for idx in range(i_total):
                vec_box[idx].m_iCol = i_max_x - vec_box[idx].m_iCol
                vec_box[idx].m_iRow = vec_box[idx].m_iRow - i_min_y
        else:  # RB
            for idx in range(i_total):
                vec_box[idx].m_iCol = i_max_x - vec_box[idx].m_iCol
                vec_box[idx].m_iRow = i_max_y - vec_box[idx].m_iRow

    def _GetBoxByRowCol(self, vec_box, ix, iy):
        ret = -1
        i_total = len(vec_box)
        for idx in range(i_total):
            if vec_box[idx].m_iCol == ix and vec_box[idx].m_iRow == iy:
                ret = idx
                break
        return ret


    def _CopyImage2Box(self, box, src_img_gray):
        i_half_w = box.imgMatch.shape[1] // 2
        i_half_h = box.imgMatch.shape[0] // 2

        i_st_x_src = box.ptCent[0] - i_half_w
        i_st_y_src = box.ptCent[1] - i_half_h
        i_end_x_src = i_st_x_src + box.imgMatch.shape[1]
        i_end_y_src = i_st_y_src + box.imgMatch.shape[0]
        i_st_x_dst = 0
        i_st_y_dst = 0
        if i_st_x_src < 0:
            i_st_x_dst = -i_st_x_src
            i_st_x_src = 0
        if i_st_y_src < 0:
            i_st_y_dst = -i_st_y_src
            i_st_y_src = 0
        if i_end_x_src > src_img_gray.shape[1]:
            i_end_x_src = src_img_gray.shape[1]
        if i_end_y_src > src_img_gray.shape[0]:
            i_end_y_src = src_img_gray.shape[0]

        src_rect = (i_st_x_src, i_st_y_src, i_end_x_src - i_st_x_src, i_end_y_src - i_st_y_src)
        dst_rect = (i_st_x_dst, i_st_y_dst, src_rect[2], src_rect[3])
        box.imgMatch[dst_rect[1]:dst_rect[1]+dst_rect[3], dst_rect[0]:dst_rect[0]+dst_rect[2]] = src_img_gray[src_rect[1]:src_rect[1]+src_rect[3], src_rect[0]:src_rect[0]+src_rect[2]]


    def _CompleteRoundBox(self, vec_box, src_img_gray, i_dir):
        i_total = len(vec_box)
        if i_total <= 0:
            return

        i_max_row = 0
        i_max_col = 0
        for idx in range(i_total):
            if vec_box[idx].m_iRow > i_max_row:
                i_max_row = vec_box[idx].m_iRow
            if vec_box[idx].m_iCol > i_max_col:
                i_max_col = vec_box[idx].m_iCol

        i_row_cnt = i_max_row + 1
        i_col_cnt = i_max_col + 1

        # 补列
        for idx in range(i_row_cnt):
            box = _CKDWater_MatchBox()
            box.m_iCol = i_col_cnt
            box.m_iRow = idx

            i_last_box_idx0 = self._GetBoxByRowCol(vec_box, i_max_col, box.m_iRow)
            i_last_box_idx1 = self._GetBoxByRowCol(vec_box, i_max_col - 1, box.m_iRow)
            if i_last_box_idx0 >= 0 and i_last_box_idx1 >= 0:
                p_box0 = vec_box[i_last_box_idx0]
                p_box1 = vec_box[i_last_box_idx1]
                box.ptCent = (np.array(p_box0.ptCent) + (np.array(p_box0.ptCent) - np.array(p_box1.ptCent))).tolist()

                box.imgMatch = p_box0.imgMatch.copy()
                box.imgBoxFull = p_box0.imgBoxFull.copy()

                self._CopyImage2Box(box, src_img_gray)

                vec_box.append(box)

        # 补行
        for idx in range(i_col_cnt+1):
            box = _CKDWater_MatchBox()
            box.m_iCol = idx
            box.m_iRow = i_row_cnt

            i_last_box_idx0 = self._GetBoxByRowCol(vec_box, box.m_iCol, i_max_row)
            i_last_box_idx1 = self._GetBoxByRowCol(vec_box, box.m_iCol, i_max_row - 1)
            if i_last_box_idx0 >= 0 and i_last_box_idx1 >= 0:
                p_box0 = vec_box[i_last_box_idx0]
                p_box1 = vec_box[i_last_box_idx1]
                box.ptCent = (np.array(p_box0.ptCent) + (np.array(p_box0.ptCent) - np.array(p_box1.ptCent))).tolist()

                box.imgMatch = p_box0.imgMatch.copy()
                box.imgBoxFull = p_box0.imgBoxFull.copy()

                self._CopyImage2Box(box, src_img_gray)

                vec_box.append(box)

    def _SearchDefect(self, vecDefect, box, srcImgGray, iDir):
        vecDefect.clear()

        vecBox = []
        vecBox.append(box)
        iIndex_R = 0
        while iIndex_R < len(vecBox):
            curBox = vecBox[iIndex_R]
            iIndex_R += 1
            #pltShowCVDot(srcImgGray, curBox.ptCent, "srcImgGray")
            self._SearchBox_Neib(vecBox, curBox, srcImgGray)

        self._UniteSortRowCol(vecBox, iDir)

        self._CompleteRoundBox(vecBox, srcImgGray, iDir)

        for idx in range(len(vecBox)):
            vecBox[idx].CalcMeanVal()
        self.vecBox = vecBox
        vecDefect_Tmp = []
        #vecDefect_Tmp.extend([_TKDWater_Defect2()] * 64)
        for idx in range(len(vecBox)):
            self._GetDefect_Neib(vecDefect_Tmp, vecBox, idx, iDir)
        if len(vecDefect_Tmp) > 0:
            vecDefect = vecDefect_Tmp

        if 0:
            imgColor = cv2.cvtColor(srcImgGray, cv2.COLOR_GRAY2BGR)
            buf = ""
            iHalfW = self.m_imgTempl.shape[1] >> 1
            iHalfH = self.m_imgTempl.shape[0] >> 1
            for idx in range(len(vecBox)):
                rect = cv2.Rect(vecBox[idx].ptCent[0] - iHalfW, vecBox[idx].ptCent[1] - iHalfH, self.m_imgTempl.shape[1], self.m_imgTempl.shape[0])
                cv2.rectangle(imgColor, rect, cv2.Scalar(0, 255, 0), 1)

                buf = str(vecBox[idx].m_iRow) + ", " + str(vecBox[idx].m_iCol)
                cv2.putText(imgColor, buf, vecBox[idx].ptCent, cv2.FONT_HERSHEY_SIMPLEX, 0.8, cv2.Scalar(0, 0, 255), 2)
            cv2.imwrite("1.jpg", imgColor)
        return vecDefect
    def _GetBoxByRowCol_NoSpecTab(self, vec_box, ix, iy):
        if (ix == 0 and iy == 3) or (ix == 0 and iy == 4) or (ix == 3 and iy == 0) or (ix == 4 and iy == 0):
            return -1

        ret = -1
        i_total = len(vec_box)
        for idx in range(i_total):
            if vec_box[idx].m_iCol == ix and vec_box[idx].m_iRow == iy:
                ret = idx
                break
        return ret

    def _GetDefect_Neib(self, vecDefect, vecBox, iCurIndex, iDir):
        if len(vecBox) < 2:
            return

        bNeib = [False, False, False, False]
        iNeibIndex = [iCurIndex, iCurIndex, iCurIndex, iCurIndex]
        iCntValid = 0

        curBox = vecBox[iCurIndex]
        iCurRow = curBox.m_iRow
        iCurCol = curBox.m_iCol
        iTotal = len(vecBox)
        curRect = (
            curBox.ptCent[0] - (self.m_imgTempl.shape[1] >> 1),
            curBox.ptCent[1] - (self.m_imgTempl.shape[0] >> 1),
            self.m_imgTempl.shape[1],
            self.m_imgTempl.shape[0])

        iIndex = self._GetBoxByRowCol_NoSpecTab(vecBox, iCurCol - 1, iCurRow)
        if iIndex >= 0:
            bNeib[0] = True
            iNeibIndex[0] = iIndex
            iCntValid += 1

        iIndex = self._GetBoxByRowCol_NoSpecTab(vecBox, iCurCol + 1, iCurRow)
        if iIndex >= 0:
            bNeib[1] = True
            iNeibIndex[1] = iIndex
            iCntValid += 1

        iIndex = self._GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow - 1)
        if iIndex >= 0:
            bNeib[2] = True
            iNeibIndex[2] = iIndex
            iCntValid += 1

        iIndex = self._GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow + 1)
        if iIndex >= 0:
            bNeib[3] = True
            iNeibIndex[3] = iIndex
            iCntValid += 1

        imgCur = vecBox[iCurIndex].imgMatch
        # pltShowCV(imgCur, "imgCur")

        DEF_THRESD_AREA = 180
        DEF_THRESD_DIFF = 60
        DEF_THRESD_NOR = 50
        DEF_THRESD_MAXMEAN = 35

        if 0:
            DEF_THRESD_DIFF_AUTO = DEF_THRESD_DIFF + int((vecBox[iCurIndex].m_iMeanVal - 80) * 0.5)
            # 创建灰度图像
            img_diff = np.full(self.m_imgTempl.shape, DEF_THRESD_DIFF_AUTO, dtype=np.uint8)
            # 画一个填充框，填充78
            cv2.rectangle(img_diff, (50, 50), (200, 200), DEF_THRESD_DIFF, -1)
            #pltShowCV(img_diff, "img_diff")
            diff = [None] * 4
            for idx in range(4):
                # pltShowCV(vecBox[iNeibIndex[idx]].imgMatch, "vecBox[iNeibIndex[idx]].imgMatch"+str(idx))
                diff1_ = cv2.absdiff(vecBox[iNeibIndex[idx]].imgMatch, imgCur)
                #pltShowCV(diff1_, "diff1_"+str(idx))
                diff_ = cv2.subtract(diff1_, img_diff)
                diff_[diff_ < 0] = 0
                # diff[idx] = cv2.absdiff(diff_, img_diff)
                _, diff[idx] = cv2.threshold(diff_, 0, 1, cv2.THRESH_BINARY)

                # diff[idx] = cv2.absdiff(vecBox[iNeibIndex[idx]].imgMatch, imgCur)
                # _, diff[idx] = cv2.threshold(diff[idx], DEF_THRESD_DIFF_AUTO, 1, cv2.THRESH_BINARY)
        else:

            diff = [None] * 4
            for idx in range(4):
                # pltShowCV(vecBox[iNeibIndex[idx]].imgMatch, "imgMatch" + str(idx))
                diff[idx] = cv2.absdiff(vecBox[iNeibIndex[idx]].imgMatch, imgCur)
                # pltShowCV(diff[idx], "diff[idx]" + str(idx))
                _, diff[idx] = cv2.threshold(diff[idx], DEF_THRESD_DIFF, 1, cv2.THRESH_BINARY)
                # pltShowCV(diff[idx]*255, "diff[idx]" + str(idx))

# NIC
        nic = [True] * 9
        for nic_idx in range(9):
            box_cur = vecBox[iCurIndex]
            nic_centerPt = self.m_vptNIC[nic_idx]
            mean_imgCur = calculate_mean(imgCur, nic_centerPt)
            mean_imgCur = box_cur.m_nic_mean[nic_idx]
            diff_cnt = 0
            for idx in range(4):
                box_Neib = vecBox[iNeibIndex[idx]]
                img_iNeibIndex = vecBox[iNeibIndex[idx]].imgMatch
                mean_NeibIndex = calculate_mean(img_iNeibIndex, nic_centerPt)
                mean_NeibIndex = box_Neib.m_nic_mean[nic_idx]
                if mean_NeibIndex - mean_imgCur >= 60:
                    diff_cnt = diff_cnt +1
            if diff_cnt > 1:
                nic_offset_x = int((3 - box_cur.m_iCol)*1.7)
                nic_offset_y = int((3 - box_cur.m_iRow)*1.7)
                nic[nic_idx] = False
                defect = _TKDWater_Defect2()
                defect.bNIC = True
                defect.ptCenter = [
                    nic_centerPt[0] + curRect[0] + nic_offset_x,
                    nic_centerPt[1] + curRect[1] + nic_offset_y]
                vecDefect.append(defect)


        for idx in range(1, 4):
            diff[0] += diff[idx]
        #pltShowCV(imgCur, "imgCur")
        #pltShowCV(diff[0]*80, "diff[0]")
        _, imgBin = cv2.threshold(diff[0], 1, 255, cv2.THRESH_BINARY)
        #pltShowCV(imgBin, "imgBin")

        _, imgBaseBin = cv2.threshold(imgCur, DEF_THRESD_NOR, 255, cv2.THRESH_BINARY_INV)
        #pltShowCV(imgBaseBin, "imgBaseBin")

        imgBin &= imgBaseBin
        # pltShowCV(imgBin, "imgBin")

        self._CheckSpecTab(imgBin, vecBox[iCurIndex], iDir)

        kernel0 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        imgBin = cv2.dilate(imgBin, kernel0)
        imgBin = cv2.erode(imgBin, kernel0)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        imgBin = cv2.erode(imgBin, kernel)
        imgBin = cv2.dilate(imgBin, kernel)

        vecCont, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for idx in range(len(vecCont)):
            area = cv2.contourArea(vecCont[idx])
            rotRect = cv2.minAreaRect(vecCont[idx])
            dMinSize = rotRect[1][0] if rotRect[1][0] < rotRect[1][1] else rotRect[1][1]

            # Calculate the moments of the contour
            moments = cv2.moments(vecCont[idx])
            # Calculate the center of the contour
            center = (moments['m10'] / moments['m00']+0.0001, moments['m01'] / moments['m00']+0.0001)
            iPos = 0  # 0: middle, 1: corner, 2: edge
            MAXMEAN = 5
            if (center[0] < 30) or (center[1] < 30) or (center[0] > 220) or (center[1] > 220):
                iPos = 2
                MAXMEAN = 0
            elif (center[0] < 50) or (center[1] < 50) or (center[0] > 200) or (center[1] > 200):
                iPos = 1
                MAXMEAN = 0


            if (area > DEF_THRESD_AREA) and (dMinSize >= 7):
                imgTmp = np.zeros(imgCur.shape[:2], dtype=np.uint8)
                cv2.drawContours(imgTmp, vecCont, idx, 255, -1)

                meanV = cv2.mean(imgCur, mask=imgTmp)[0]
                if meanV < DEF_THRESD_MAXMEAN + MAXMEAN:
                    # 判断
                    defect = _TKDWater_Defect2()
                    defect.iArea = area
                    defect.rtRect = cv2.boundingRect(vecCont[idx])
                    defect.iPos = iPos
                    defect.rtRect = [
                        defect.rtRect[0] + curRect[0],
                        defect.rtRect[1] + curRect[1], defect.rtRect[2], defect.rtRect[3]]
                    vecDefect.append(defect)

    def _CheckSpecTab(self, imgBin, curBox, iDir):
        imgtabMask = None

        if iDir == 0: # LT
            if curBox.m_iCol == 3 and curBox.m_iRow == 0:
                imgtabMask = self.m_imgTempl_tabMask_Col
                imgBin &= imgtabMask
            elif curBox.m_iCol == 4 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, 1)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 3:
                imgtabMask = self.m_imgTempl_tabMask_Row
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 4:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, 0)
                imgBin &= imgtabMask

        elif iDir == 1:  # LB
            if curBox.m_iCol == 3 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, 0)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 4 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, -1)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 3:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, 0)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 4:
                imgtabMask = self.m_imgTempl_tabMask_Row
                imgBin &= imgtabMask

        if iDir == 2:  # RT
            if curBox.m_iCol == 3 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, 1)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 4 and curBox.m_iRow == 0:
                imgtabMask = self.m_imgTempl_tabMask_Col
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 3:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, 1)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 4:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, -1)
                imgBin &= imgtabMask

        else:  # RB
            if curBox.m_iCol == 3 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, 1)
                imgtabMask = cv2.flip(imgtabMask, 0)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 4 and curBox.m_iRow == 0:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Col, 0)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 3:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, -1)
                imgBin &= imgtabMask
            elif curBox.m_iCol == 0 and curBox.m_iRow == 4:
                imgtabMask = cv2.flip(self.m_imgTempl_tabMask_Row, 1)
                imgBin &= imgtabMask

    def _CheckDefect_ByRound(self, vecDefect, vecDefect_Tmp, srcImgGray):
        iTotal = len(vecDefect_Tmp)

        for idx in range(iTotal):
            curRect = vecDefect_Tmp[idx].rtRect
            iCurMeanVal = cv2.mean(srcImgGray(curRect))[0]

            bValid = [False, False, False, False]
            iSubMeanVal = [0, 0, 0, 0]

            subRect = curRect
            subRect[0] -= curRect[2]
            if subRect[0] >= 0:
                bValid[0] = True
                iSubMeanVal[0] = cv2.mean(srcImgGray(subRect))[0]

            subRect = curRect
            subRect[0] += curRect[2]
            if subRect[0] + subRect[2] <= srcImgGray.shape[1]:
                bValid[1] = True
                iSubMeanVal[1] = cv2.mean(srcImgGray(subRect))[0]

            subRect = curRect
            subRect[1] -= curRect[3]
            if subRect[1] >= 0:
                bValid[2] = True
                iSubMeanVal[2] = cv2.mean(srcImgGray(subRect))[0]

            subRect = curRect
            subRect[1] += curRect[3]
            if subRect[1] + subRect[3] <= srcImgGray.shape[0]:
                bValid[3] = True
                iSubMeanVal[3] = cv2.mean(srcImgGray(subRect))[0]

            iSumCnt = 0
            for tt in range(4):
                if bValid[tt]:
                    if iCurMeanVal < iSubMeanVal[tt] - 30:
                        iSumCnt += 1
            if iSumCnt > 0:
                vecDefect.append(vecDefect_Tmp[idx])

    def _SearchBox_Neib(self, vecBox, curBox, srcImgGray):
        self._GetNeibBox(curBox, [-self.m_ptStdOffs[0], 0], vecBox, srcImgGray)  # Left
        self._GetNeibBox(curBox, (self.m_ptStdOffs[0], 0), vecBox, srcImgGray)   # Right
        self._GetNeibBox(curBox, (0, -self.m_ptStdOffs[1]), vecBox, srcImgGray)  # Up
        self._GetNeibBox(curBox, (0, self.m_ptStdOffs[1]), vecBox, srcImgGray)   # Down

    def _GetNeibBox(self, curBox, offsPt, vecBox, srcImgGray):
        iIndex = -1
        try:
            dstCent = np.array(curBox.ptCent) + offsPt
        except:
            pass
        for idx in range(len(vecBox)):
            if vecBox[idx].IsPtInRect(dstCent):
                iIndex = idx
                break

        if iIndex >= 0:
            return True

        iMatchBoxHalfWid = (self.m_imgTempl.shape[1] >> 1) + 30
        iMatchBoxHalfHei = (self.m_imgTempl.shape[0] >> 1) + 30
        iMatchBoxWid = iMatchBoxHalfWid << 1
        iMatchBoxHei = iMatchBoxHalfHei << 1
        ret = False

        srcRect = (dstCent[0] - iMatchBoxHalfWid, dstCent[1] - iMatchBoxHalfHei, iMatchBoxWid, iMatchBoxHei)

        outBox = _CKDWater_MatchBox()

        if (srcRect[0] >= 0) and (srcRect[1] >= 0) and (srcRect[0] + srcRect[2] < srcImgGray.shape[1]) and (srcRect[1] + srcRect[3] < srcImgGray.shape[0]):
            matchImg = srcImgGray[srcRect[1]:srcRect[1] + srcRect[3], srcRect[0]:srcRect[0] + srcRect[2]].copy()
            if self._GetMaxMatch(outBox, matchImg, curBox.imgMatch):
                outBox.ptCent[0] += srcRect[0]
                outBox.ptCent[1] += srcRect[1]
                ret = True

        if ret:
            outBox.m_iCol = curBox.m_iCol + (1 if offsPt[0] > 0 else (-1 if offsPt[0] < 0 else 0))
            outBox.m_iRow = curBox.m_iRow + (1 if offsPt[1] > 0 else (-1 if offsPt[1] < 0 else 0))
            vecBox.append(outBox)

        return ret

    def _GetMaxMatch(self, outBox, roiImg, imgTempl):
        imgTempl_Sub = cv2.pyrDown(imgTempl)
        imgRoi_Sub = cv2.pyrDown(roiImg)

        result = cv2.matchTemplate(imgRoi_Sub, imgTempl_Sub, cv2.TM_CCOEFF_NORMED)
        result = cv2.convertScaleAbs(result, alpha=(255.0))
        dMax = 0
        _, dMax, _, maxLoc = cv2.minMaxLoc(result)

        iVal = dMax
        if iVal < 150:
            return False

        iSumX = 0
        iSumY = 0
        cnt = 0
        for row in range(result.shape[0]):
            pLine = result[row]
            for col in range(result.shape[1]):
                if pLine[col] >= iVal:
                    iSumX += col
                    iSumY += row
                    cnt += 1
        if cnt <= 0:
            return False

        outBox.ptCent[0] = int(iSumX * 2 / cnt)
        outBox.ptCent[1] = int(iSumY * 2 / cnt)

        roiRect = (outBox.ptCent[0], outBox.ptCent[1], imgTempl.shape[1], imgTempl.shape[0])
        outBox.imgMatch = roiImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]].copy()
        # roiImg[roiRect[1]:roiRect[1] + roiRect[3], roiRect[0]:roiRect[0] + roiRect[2]].copyTo(outBox.imgMatch)
        # roiImg.copyTo(outBox.imgBoxFull)
        outBox.imgBoxFull = roiImg.copy()

        outBox.ptCent[0] += imgTempl.shape[1] >> 1
        outBox.ptCent[1] += imgTempl.shape[0] >> 1

        return True

    def _GetBestMatchPos(self, ptCent, srcImgGray):
        result = cv2.matchTemplate(srcImgGray, self.m_imgTempl, cv2.TM_CCOEFF_NORMED)
        result = cv2.convertScaleAbs(result, alpha=(255.0))

        dMax = 0
        _, dMax, _, maxLoc = cv2.minMaxLoc(result)

        dMax -= 1.0
        if dMax < 100:
            dMax = 100

        imgBin = cv2.threshold(result, dMax, 255, cv2.THRESH_BINARY)[1]

        vecCont, _ = cv2.findContours(imgBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
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

def delete_path(path):
    # Delete files in the path
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Delete folders in the path
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        if os.path.isdir(file_path):
            os.rmdir(file_path)

def CheckWater_Common(filePath, strResult, iDir):
    global iDir_g
    iDir_g = iDir
    # Delete the existing result folder
    if os.path.exists(strResult):
        for file in os.listdir(strResult):
            os.remove(os.path.join(strResult, file))
        os.rmdir(strResult)

    # Create the result folder
    os.mkdir(strResult)

    # Scan the files in the given path
    vecPath = []
    for file in os.listdir(filePath):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".bmp"):
            vecPath.append(file)

    # Load the template
    algo = CAlgo_KDWater2()
    algo.LoadTemplate()

    # Find defects in each image and save the result
    for idx in range(len(vecPath)):
        img = cv2.imread(os.path.join(filePath, vecPath[idx]), cv2.IMREAD_GRAYSCALE)
        imgColor = cv2.imread(os.path.join(filePath, vecPath[idx]))
        vecDefect = []
        result, vecDefect = algo.FindDefect(vecDefect, img, iDir)
# 画圈
        for box in algo.vecBox:
            #center_nic = [box.ptCent[0] + (box.m_iCol-3)*2, box.ptCent[1] + (box.m_iRow-3)*2]
            zero = 1.85
            radio = 2.42
            if iDir == 0:
                nic_offset_x = int((zero - box.m_iCol)*radio)
                nic_offset_y = int((zero - box.m_iRow)*radio)
            elif iDir == 1:
                zero = 2.5
                radio = 1.8
                nic_offset_x = int((zero - box.m_iCol)*radio)
                nic_offset_y = int((box.m_iRow-zero)*radio)
            elif iDir == 2:
                zero = 2.5
                radio = 1.8
                nic_offset_x = int((box.m_iCol - zero)*radio)
                nic_offset_y = int((zero - box.m_iRow)*radio)
            else:
                zero = 2.5
                radio = 1.8
                nic_offset_x = int((box.m_iCol - zero)*radio)
                nic_offset_y = int((box.m_iRow-zero)*radio)

            # nic_offset_x = 0
            # nic_offset_y = 0
            lt = [box.ptCent[0]-124, box.ptCent[1]-124]

            rect = [box.ptCent[0] - 124, box.ptCent[1] - 124, 248, 248]
            #cv2.rectangle(imgColor, rect, (0, 255, 0), 1)
            for i in range(9):
                nic_center = box.m_vptNIC[i]
                nic_mean = box.m_nic_mean[i]
                nic_center_new = [nic_center[0] + lt[0] + nic_offset_x, nic_center[1] + lt[1]+nic_offset_y]
                cv2.circle(imgColor, nic_center_new, 18, (0, 255, 0), 1)
                cv2.putText(imgColor, str(int(nic_mean)), [nic_center[0] + lt[0]-15, nic_center[1] + lt[1]-15], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        filename = os.path.join(strResult, vecPath[idx])
        cv2.imwrite(filename, imgColor)
        return
        if not result:

            for idx1 in range(len(vecDefect)):
                if vecDefect[idx1].bNIC:
                    cv2.circle(imgColor, vecDefect[idx1].ptCenter, 15, (255, 0, 0), 1)
                else:
                    boxRT = vecDefect[idx1].rtRect
                    boxRT[0] -= 0
                    boxRT[1] -= 0
                    boxRT[2] += 0
                    boxRT[3] += 0
                    if boxRT[0] < 0: boxRT[0] = 0
                    if boxRT[1] < 0: boxRT[1] = 0
                    if boxRT[0] + boxRT[2] > imgColor.shape[1]: boxRT[2] = imgColor.shape[1] - boxRT[0]
                    if boxRT[1] + boxRT[3] > imgColor.shape[0]: boxRT[3] = imgColor.shape[0] - boxRT[1]

                    #cv2.rectangle(imgColor, boxRT, (0, 0, 255), 1)

                    boxRT_tl = [boxRT[0], boxRT[1]]
                    #cv2.putText(imgColor, str(int(vecDefect[idx1].iArea)), boxRT_tl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)

            filename = os.path.join(strResult, vecPath[idx])
            cv2.imwrite(filename, imgColor)

def CheckWater_1_1():
    filePath = dir_root + "1_1"
    strResult = dir_root + "Result_1_1"
    CheckWater_Common(filePath, strResult, 0)

def CheckWater_1_2():
    filePath = dir_root + "1_2"
    strResult = dir_root + "Result_1_2"
    CheckWater_Common(filePath, strResult, 1)

def CheckWater_2_1():
    filePath = dir_root + "2_1"
    strResult = dir_root + "Result_2_1"
    CheckWater_Common(filePath, strResult, 2)

def CheckWater_2_2():
    filePath = dir_root + "2_2"
    strResult = dir_root + "Result_2_2"
    CheckWater_Common(filePath, strResult, 3)
iDir_g = 0
if __name__ == '__main__':
    dir_root = r"D:\0\0LG_DATA\SZ_NIC_4/"
    CheckWater_1_1()
    CheckWater_1_2()
    CheckWater_2_1()
    CheckWater_2_2()
