# 导入必要的库
import cv2
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
    def __init__(self, p1: Tuple[int, int], p2: Tuple[int, int], area: int, type: int):
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
        self.m_img1Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("1c2", "cb5", 2, (700, CTOP), (1000, CBOTTOM), AREA150),
            CBox("1c3", "cb1", 3, (1000, CTOP), (1430, CBOTTOM)),
            CBox("1c4", "cb5", 4, (1900, CTOP), (2250, CBOTTOM), AREA150),
            CBox("1b6", "bbl", 6, (450, BTOP), (1430, BBOTTOM), AREA25),
            CBox("1b8", "bbl", 8, (1430, BTOP), (2250, BBOTTOM), AREA150),
            CBox("1b2", "bbc", 2, (2000, BTOP), (2448, BBOTTOM))
        ]
        self.m_img2Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("2c5", "cb5", 5, (600, CTOP), (1200, CBOTTOM), AREA150),
            CBox("2c6", "cb2", 6, (1200, CTOP), (1800, CBOTTOM)),
            CBox("2b1", "bbc", 1, (0, BTOP), (2448, BBOTTOM))
        ]
        self.m_img3Cfg = [
            CBox("1c1", "cb5", 1, (450, CTOP), (700, CBOTTOM), AREA150),
            CBox("3c7", "cb3", 7, (300, CTOP), (900, CBOTTOM)),
            CBox("3c8", "cb6", 8, (1000, CTOP), (1600, CBOTTOM), AREA150),
            CBox("3c9", "cb6", 9, (2000, CTOP), (2448, CBOTTOM), AREA150),
            CBox("3b3", "bbc", 1, (0, BTOP), (2448, BBOTTOM))
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
        self.m_brokenCfg = {"cb1": 0, "cb2": 0, "cb3": 0, "cb4": 0, "cb5": 1, "cb6": 1, "bbl": 1, "bbc": 0, "bbr": 1}
        self.m_brokenCnt = {}
        self.m_objs = CBoxArray()

    # v_img 四张图片
    # vv_defect 四个CDefect缺陷列表
    def Process(self, v_img: List[np.ndarray], vv_defect: List[List[CDefect]]) -> bool:
        pass

    def processImg1(self, img: np.ndarray, defect: CDefect, serial: int):
        pass

    def getMask(self, points: List[Tuple[int, int]]) -> np.ndarray:
        pass

if __name__ == '__main__':
    dir_root = "F:/sheepy/02data/01LG/test/"
    img_first_name = "black_0074690_CM1_"
    v_img = []
    for i in range(4):
        img_path = dir_root + img_first_name + str(i + 1) + ".bmp"
        v_img.append(cv2.imread(img_path))

    pp = CPostProcessor()