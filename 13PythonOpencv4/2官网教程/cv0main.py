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
cv6HoughLineTransform = cv2Transformations.cv6HoughLineTransform            # 霍夫直线检测
#from cv3ImageProcessing.cv2Transformations import cv6HoughLineTransform