from cv3ImageProcessing import cv1Basic
# 1绘图
my_ellipse = cv1Basic.cv1BasicDrawing.my_ellipse                                # 画椭圆
my_circle = cv1Basic.cv1BasicDrawing.my_circle                                  # 画圆
my_fillPoly = cv1Basic.cv1BasicDrawing.my_fillPoly                              # 填充轮廓
my_line = cv1Basic.cv1BasicDrawing.my_line                                      # 画线
my_rectangle = cv1Basic.cv1BasicDrawing.my_rectangle                            # 画框
# 2随机绘图
Drawing_Random_Lines = cv1Basic.cv2RandomDrawing.Drawing_Random_Lines           # 随机画线
Drawing_Random_rect = cv1Basic.cv2RandomDrawing.Drawing_Random_rect             # 随机画框
Drawing_Random_circle = cv1Basic.cv2RandomDrawing.Drawing_Random_circle         # 随机画圆
# 3平滑图像
my_blur = cv1Basic.cv3SmoothingImages.my_blur                                   # 标准模糊
my_GaussianBlur = cv1Basic.cv3SmoothingImages.my_GaussianBlur                   # 高斯模糊
my_medianBlur = cv1Basic.cv3SmoothingImages.my_medianBlur                       # 中位数模糊
# 4侵蚀和膨胀
cv4ErodingAndDilating = cv1Basic.cv4ErodingAndDilating                          # 侵蚀和膨胀
# 5开闭和礼帽运算
cv5MoreMorphologyTransformations = cv1Basic.cv5MoreMorphologyTransformations
my_opening = cv5MoreMorphologyTransformations.my_opening                        # 开运算
my_closing = cv5MoreMorphologyTransformations.my_closing                        # 闭运算
my_gradient = cv5MoreMorphologyTransformations.my_gradient                      # 梯度 轮廓
my_tophat = cv5MoreMorphologyTransformations.my_tophat                          # 礼帽 input-opening
my_blackhat = cv5MoreMorphologyTransformations.my_blackhat                      # 黑帽 input-closing
# 6击中-击不中（匹配）
cv6HitMiss = cv1Basic.cv6HitMiss
# 7形态转换检测线
cv7MorphLinesDetection = cv1Basic.cv7MorphLinesDetection
# 8图像金字塔
cv8ImagePyramids = cv1Basic.cv8ImagePyramids
# 9阈值基本操作
cv9BasicThresholdingOperations = cv1Basic.cv9BasicThresholdingOperations
# 10使用inRange进行阈值操作
cv10ThresholdUseInRange = cv1Basic.cv10ThresholdUseInRange

from cv3ImageProcessing import cv2Transformations
cv6HoughLineTransform = cv2Transformations.cv6HoughLineTransform                # 霍夫直线检测
# 10仿射变换
cv10AffineTransformations = cv2Transformations.cv10AffineTransformations
my_warpAffine = cv10AffineTransformations.my_warpAffine                         # 仿射变换
myRotation_warpAffine = cv10AffineTransformations.myRotation_warpAffine         # 旋转仿射变换

from cv3ImageProcessing import cv3Histograms
# 直方图均衡化
cv1HistogramEqualization = cv3Histograms.cv1HistogramEqualization
my_equalizeHist = cv1HistogramEqualization.my_equalizeHist                      # 直方图均衡化
myBGR_equalizeHist = cv1HistogramEqualization.myBGR_equalizeHist                # 分通道直方图均衡化
# 计算直方图
cv2HistogramCalculation = cv3Histograms.cv2HistogramCalculation
myGrayList_calcHist = cv2HistogramCalculation.myGrayList_calcHist               # 计算直方图-灰度图列表
mybgr_calcHist = cv2HistogramCalculation.mybgr_calcHist                         # 计算直方图-彩色图列表
# 直方图比较
my_compareHist = cv3Histograms.cv3HistogramComparison.my_compareHist            # 直方图比较
# 背投影
my_floodFill = cv3Histograms.cv4CalcBackProjection.my_floodFill                 # 漫水填充(涂油漆桶)
my_calcBackProject = cv3Histograms.cv4CalcBackProjection.my_calcBackProject     # 背投影 点在直方图的概率值
# 模板匹配
my_matchTemplate = cv3Histograms.cv4CalcBackProjection.my_matchTemplate         # 模板匹配

from cv3ImageProcessing import cv4Contours
my_findContours = cv4Contours.cv1findContours.my_findContours                   # 寻找轮廓
my_convexHull = cv4Contours.cv2ConvexHull.my_convexHull                       # 凸包


