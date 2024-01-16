import cv2
import numpy as np

class TImgProc_DefectInspParam:
    def __init__(self):
        self.iMinArea = 0
        self.iMinHei = 0
        self.iMinWid = 0

class TImgProc_DefectParam:
    def __init__(self):
        self.iThresd = 0
        self.iKnifeNum = 0
        self.iKnifePos = [[0, 0] for _ in range(6)]
        self.iRetract_WB = 0
        self.iRetract_SubWB = 0
        self.iSubDefectImgSize = 0
        self.dRateWH = 0.0
        self.iDefectPixel_Threshold = [0, 0, 0, 0]
        self.tDefectParamArr = [TImgProc_DefectInspParam() for _ in range(6)]

class CImgProc_Blob:
    def __init__(self):
        self.iType = 0
        self.iArea = 0
        self.iPos_X = 0
        self.iPos_Y = 0
        self.iSize_Wid = 0
        self.iSize_Hei = 0
        self.iSegIdx = 0
        self.rcRect = []
        self.iPixelValue = [0, 0, 0]
        self.imgDefect = None

class CImgProc_Result:
    def __init__(self):
        self.Reset()

    def Reset(self):
        self.m_iResult_ImgInsp = 0
        self.m_iEdgeX = [0.0, 0.0]
        self.m_vecBlob = []