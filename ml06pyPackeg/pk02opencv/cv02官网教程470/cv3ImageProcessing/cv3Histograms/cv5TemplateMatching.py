import sys
import cv2 as cv

# 模板匹配
def my_matchTemplate(img_bgr, tmpl_bgr, method=cv.TM_SQDIFF_NORMED, mask=None):
    # def matchTemplate(image, templ, method, result=None, mask=None)
    #
    # cv2.TM_SQDIFF         采用<平方差>的方法，求模板和图像之间的差值，差值越小匹配度越高。
    # cv2.TM_SQDIFF_NORMED  采用归一化平方差的方法，将模板和图像归一化后再求平方差，差值越小匹配度越高。
    # cv2.TM_CCORR          采用<互相关>的方法，将模板和图像卷积后再求相关，相关值越大匹配度越高。
    # cv2.TM_CCORR_NORMED
    # cv2.TM_CCOEF          采用<相关系数>的方法，将模板和图像转化为概率分布后再求相关系数，相关系数越大匹配度越高。
    # cv2.TM_CCOEFF_NORMED
    result = cv.matchTemplate(img_bgr, tmpl_bgr, method, result=None, mask=mask)
    cv.normalize( result, result, 0, 1, cv.NORM_MINMAX, -1 )
    _minVal, _maxVal, minLoc, maxLoc = cv.minMaxLoc(result, None)
    if match_method == cv.TM_SQDIFF or match_method == cv.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    return matchLoc, result


def main():
    cv.namedWindow( image_window, cv.WINDOW_AUTOSIZE )
    cv.namedWindow( result_window, cv.WINDOW_AUTOSIZE )
    trackbar_label = 'Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED'
    cv.createTrackbar( trackbar_label, image_window, match_method, 5, MatchingMethod )
    MatchingMethod(match_method)
    cv.waitKey(0)
    return 0

def MatchingMethod(param):
    global match_method
    match_method = param

    img_display = img_bgr.copy()
    method_accepts_mask = (cv.TM_SQDIFF == match_method or match_method == cv.TM_CCORR_NORMED)
    if (use_mask and method_accepts_mask):
        matchLoc, result = my_matchTemplate(img_bgr, tmpl_bgr, method=match_method, mask=mask_bgr)
    else:
        matchLoc, result = my_matchTemplate(img_bgr, tmpl_bgr, method=match_method)
    cv.rectangle(img_display, matchLoc, (matchLoc[0] + tmpl_bgr.shape[0], matchLoc[1] + tmpl_bgr.shape[1]), (0, 0, 0), 2, 8, 0)
    cv.rectangle(result, matchLoc, (matchLoc[0] + tmpl_bgr.shape[0], matchLoc[1] + tmpl_bgr.shape[1]), (0, 0, 0), 2, 8, 0)
    cv.imshow(image_window, img_display)
    cv.imshow(result_window, result)
    pass

if __name__ == "__main__":
    dir_root = r"D:\02dataset\02opencv_data/"
    filename = dir_root + 'lena_tmpl.jpg'
    templatename = dir_root + 'tmpl.png'
    maskname = dir_root + 'mask.png'
    img_bgr = cv.imread(filename)
    tmpl_bgr = cv.imread(templatename)
    mask_bgr = None
    use_mask = True
    if use_mask:
        mask_bgr = cv.imread(maskname)
    image_window = "Source Image"
    result_window = "Result window"

    match_method = 0
    main()
