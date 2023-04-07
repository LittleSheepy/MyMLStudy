# 导入必要的库
import cv2
import cv2 as cv
import numpy as np
from typing import List, Tuple

CTOP = 700
CBOTTOM = 1630
BTOP = 1500
BBOTTOM = 1800
PIX_H = 1233
MM_H = 84
PIX_MM = PIX_H/MM_H
AREA25 = int(PIX_MM*PIX_MM*25)
AREA150 = int(PIX_MM*PIX_MM*150)

# 瑕疵结构体
class CDefect:
    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], area: int, type: int=0):
        self.p1 = p1    # 左上点
        self.p2 = p2    # 右下点
        self.area = area    # 缺陷面积
        self.type = type    # 缺陷类型 破损、毛边


# 瑕疵结构体
class CArrayCfg:
    def __init__(self, cnt: int = 0, area: int = 0):
        self.cnt = cnt    # 允许缺陷个数
        self.area = area    # 允许缺陷面积


class CBox:
    def __init__(self, name: str, arr_name: str, serial: int, p1: Tuple[int, int], p2: Tuple[int, int], area: int = 0):
        self.name = name
        self.arr_name = arr_name
        self.serial = serial
        self.p1 = p1
        self.p2 = p2
        self.area = area
        self.n_defect = 0
        self.state = True


class CBoxArray:
    def __init__(self):
        self.v_obj = []
        self.state = 0


class CPostProcessor:
    def __init__(self):
        self.m_img1Cfg = None
        self.m_img2Cfg = None
        self.m_img3Cfg = None
        self.m_img4Cfg = None
        self.imgCfgInit()
        self.m_imgCfg = [self.m_img1Cfg, self.m_img2Cfg, self.m_img3Cfg, self.m_img4Cfg]
        self.m_brokenCfg = {"cb1": 0, "cb2": 0, "cb3": 0, "cb4": 0, "cb5": 1, "cb6": 1, "bbl": 1, "bbc": 0, "bbr": 1}
        self.m_brokenCnt = {}
        self.m_objs = CBoxArray()
        self.offset = 0
        self.template = cv2.imread(r"D:\04DataSets\ningjingLG\/template.bmp")
        self.template_x = 1260

    # v_img 四张图片
    # vv_defect 四个CDefect缺陷列表
    def Process(self, v_img: List[np.ndarray], vv_defect: List[List[CDefect]]) -> bool:
        result = True
        # 设置offset
        self.setOffSet(v_img[0])
        self.m_brokenCnt = {}
        # 遍历4个图
        for i in range(4):
            img = v_img[i]
            v_defect = vv_defect[i]
            for defect in v_defect:
                self.processImg(img, defect, i)

        # 遍历m_brokenCnt 确认 NG
        for key, val in self.m_brokenCnt.items():
            if val > self.m_brokenCfg[key]:
                result = False
                break
        return result

    def processImg(self, img: np.ndarray, defect: CDefect, serial: int):
        img_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.rectangle(img_mask, defect.p1, defect.p2, 1, -1, 4)
        defect_area = defect.area
        for cfg in self.m_imgCfg[serial]:
            arr_name = cfg.arr_name
            cfg_area = cfg.area
            # 切片
            x1 = cfg.p1[0]+self.offset
            x2 = cfg.p2[0]+self.offset
            if x1 < 0:
                x1 = 0
            if x2 < 0:
                x2 = 0
            if x1 > img.shape[1]:
                x1 = img.shape[1]
            if x2 > img.shape[1]:
                x2 = img.shape[1]
            select = slice(cfg.p1[1], cfg.p2[1]), slice(x1, x2)
            ROI = img_mask[select]
            sum = cv2.sumElems(ROI)[0]
            # 一多半在这个配置框就认为是这个的
            if sum > defect.area*0.5:
                # 面积超限 算两个
                self.m_brokenCnt[arr_name] = self.m_brokenCnt.get(arr_name, 0)
                if defect_area > cfg_area:
                    cfg.state = False
                    cfg.n_defect += 1
                    self.m_brokenCnt[arr_name] += 1
                cfg.n_defect += 1
                self.m_brokenCnt[arr_name] += 1

    def findWhiteArea(self, img_bgr: np.ndarray):
        method = cv2.TM_SQDIFF_NORMED
        result = cv2.matchTemplate(img_bgr, self.template, method)
        cv2.normalize(result, result, 0, 1, cv.NORM_MINMAX, -1)
        _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
        if method == cv.TM_SQDIFF or method == cv.TM_SQDIFF_NORMED:
            matchLoc = minLoc
        else:
            matchLoc = maxLoc
        return matchLoc

    def setOffSet(self, img_bgr):
        matchLoc = self.findWhiteArea(img_bgr)
        self.offset = matchLoc[0] - self.template_x
        if abs(self.offset) > 250:
            self.offset = 0
            self.imgCfgInit()
        else:
            self.imgCfgInitByOffSet()

    def imgCfgInit(self):
        self.m_img1Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("1c2", "cb5", 2, (700, CTOP), (1000, CBOTTOM), AREA150),
            CBox("1c3", "cb1", 3, (1000, CTOP), (1430, CBOTTOM)),
            CBox("1c4", "cb5", 4, (1900, CTOP), (2250, CBOTTOM), AREA150),

            CBox("1b6", "bbl", 6, (430, BTOP), (1130, BBOTTOM), AREA25),
            CBox("1b8", "bbl", 8, (1200, BTOP), (2050, BBOTTOM), AREA150),
            CBox("1b2", "bbc", 2, (2150, BTOP), (2448, BBOTTOM))
        ]
        self.m_img2Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("2c5", "cb5", 5, (600, CTOP), (1200, CBOTTOM), AREA150),
            CBox("2c6", "cb2", 6, (1200, CTOP), (1800, CBOTTOM)),

            CBox("2b1", "bbc", 1, (100, BTOP), (2448, BBOTTOM))
        ]
        self.m_img3Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("3c7", "cb3", 7, (300, CTOP), (900, CBOTTOM)),
            CBox("3c8", "cb6", 8, (1000, CTOP), (1600, CBOTTOM), AREA150),
            CBox("3c9", "cb6", 9, (2000, CTOP), (2448, CBOTTOM), AREA150),
            CBox("3b3", "bbc", 1, (0, BTOP), (2200, BBOTTOM))
        ]
        self.m_img4Cfg = {
            CBox("4c9", "cb6", 9, (100, CTOP), (600, CBOTTOM), AREA150),
            CBox("4c10", "cb4", 10, (1100, CTOP), (1400, CBOTTOM)),
            CBox("4c11", "cb6", 11, (1400, CTOP), (1700, CBOTTOM), AREA150),
            CBox("4c12", "cb6", 12, (1800, CTOP), (2100, CBOTTOM), AREA150),

            CBox("4b3", "bbc", 1, (0, BTOP), (400, BBOTTOM)),
            CBox("4b9", "bbr", 9, (100, BTOP), (1400, BBOTTOM), AREA150),
            CBox("4b10", "bbr", 10, (1100, BTOP), (2100, BBOTTOM), AREA25),
        }
        self.m_imgCfg = [self.m_img1Cfg, self.m_img2Cfg, self.m_img3Cfg, self.m_img4Cfg]

    def imgCfgInitByOffSet(self):
        self.m_img1Cfg = [
            CBox("1c1", "cb5", 1, (500, CTOP), (630, CBOTTOM), AREA150),
            CBox("1c2", "cb5", 2, (840, CTOP), (920, CBOTTOM), AREA150),
            CBox("1c3", "cb1", 3, (1110, CTOP), (1200, CBOTTOM)),
            CBox("1c4", "cb5", 4, (2020, CTOP), (2150, CBOTTOM), AREA150),

            CBox("1b6", "bbl", 6, (450, BTOP), (1430, BBOTTOM), AREA25),
            CBox("1b8", "bbl", 8, (1430, BTOP), (2250, BBOTTOM), AREA150),
            CBox("1b2", "bbc", 2, (2000, BTOP), (2448, BBOTTOM))
        ]
        self.m_img2Cfg = [
            CBox("2c4", "cb5", 4, (0, CTOP), (180, CBOTTOM), AREA150),
            CBox("2c5", "cb5", 5, (820, CTOP), (960, CBOTTOM), AREA150),
            CBox("2c6", "cb2", 6, (1640, CTOP), (1800, CBOTTOM)),

            CBox("2b1", "bbc", 1, (0, BTOP), (2448, BBOTTOM))
        ]
        self.m_img3Cfg = [
            CBox("3c7", "cb3", 7, (500, CTOP), (700, CBOTTOM)),
            CBox("3c8", "cb6", 8, (1300, CTOP), (1500, CBOTTOM), AREA150),
            CBox("3c9", "cb6", 9, (2100, CTOP), (2300, CBOTTOM), AREA150),

            CBox("3b3", "bbc", 1, (0, BTOP), (2448, BBOTTOM))
        ]
        self.m_img4Cfg = {
            CBox("4c9", "cb6", 9, (250, CTOP), (420, CBOTTOM), AREA150),
            CBox("4c10", "cb4", 10, (1200, CTOP), (1300, CBOTTOM)),
            CBox("4c11", "cb6", 11, (1480, CTOP), (1560, CBOTTOM), AREA150),
            CBox("4c12", "cb6", 12, (1800, CTOP), (2030, CBOTTOM), AREA150),

            CBox("4b3", "bbc", 1, (0, BTOP), (280, BBOTTOM)),
            CBox("4b9", "bbr", 9, (380, BTOP), (1240, BBOTTOM), AREA150),
            CBox("4b10", "bbr", 10, (1300, BTOP), (2030, BBOTTOM), AREA25),
        }
        self.m_imgCfg = [self.m_img1Cfg, self.m_img2Cfg, self.m_img3Cfg, self.m_img4Cfg]

    def getMask(self, points: List[Tuple[int, int]]) -> np.ndarray:
        pass

# cv2.imwrite("img_canny.jpg", img_canny)
def mask2defectList(img_mask):
    img_canny = cv.Canny(img_mask, 127, 127 * 2)
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    defectList = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c, True)
        area = abs(area)
        boundRect = cv.boundingRect(cv.approxPolyDP(c, 3, True))
        defectList.append(CDefect(
            (boundRect[0], boundRect[1]),
            (boundRect[0] + boundRect[2], boundRect[1] + boundRect[3]), area))
    return defectList

def drawDefectList(img_bgr, defectList):
    for defect in defectList:
        cv2.rectangle(img_bgr, defect.p1, defect.p2, (255, 0, 0))
    return defectList


if __name__ == '__main__':
    dir_root = r"D:\04DataSets\ningjingLG\all\/"
    img_first_name = "black_0074690_CM1_"
    v_img = []
    for i in range(4):
        img_path = dir_root + img_first_name + str(i + 1) + ".bmp"
        v_img.append(cv2.imread(img_path))
    vv_defect = [[CDefect((450, 700), (700, 1630), 6)]] * 4

    pp = CPostProcessor()
    try:
        result = pp.Process(v_img, vv_defect)
        print("\n",result)
    except Exception as e:
        print(e)
