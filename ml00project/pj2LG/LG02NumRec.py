import cv2
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random as rng
# import sys
# sys.path.append(r'D:\03GitHub\00myGitHub\MyMLStudy\ml06pyPackeg\pk02opencv\cv02官网教程470\cv3ImageProcessing\cv4Contours/')
# from ...ml06pyPackeg.pk02opencv.cv02官网教程470.cv3ImageProcessing.cv4Contours import cv1findContours
#


def draw_contours(img_gray, contours, hierarchy):
    drawing = np.zeros((img_gray.shape[0], img_gray.shape[1], 3), dtype=np.uint8)
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2, cv.LINE_8, hierarchy, 0)
    return drawing

def save_num_line(num_img, white_x_min, white_x_max):
    num_img = num_img.copy()
    num_img = cv2.cvtColor(num_img, cv2.COLOR_GRAY2BGR)
    for index in range(7):
        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        xmin_ = white_x_min[index]
        xmax_ = white_x_max[index]
        cv2.line(num_img, (xmin_, 0), (xmin_, num_img.shape[0]), color, 1)
        cv2.line(num_img, (xmax_, 0), (xmax_, num_img.shape[0]), color, 1)
    cv2.imwrite(imgNumLine_file_path, num_img)


def imgNumArea(num_img, binary_img):
    num_img_h, num_img_w = num_img.shape[0], num_img.shape[1]
    ratio = num_img_h/35
    num_img_w_new = int(num_img_w / ratio)
    num_img = cv2.resize(num_img, (num_img_w_new, 35))
    binary_img = cv2.resize(binary_img, (num_img_w_new, 35))
    num_img_org = num_img.copy()
    num_img = cv2.cvtColor(num_img, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    for i, contour in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

        color = (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256))
        cv.drawContours(num_img, contours, i, color, 1, cv.LINE_8, hierarchy, 0)
        # center_x = int(boundRect[i][0] + boundRect[i][2]/2)
        # center_y = int(boundRect[i][1] + boundRect[i][3]/2)
        # w, h = 20, 32
        # point1 = (center_x-w//2, center_y - h//2)
        # point2 = (center_x+w//2, center_y + h//2)
        # num_img = cv2.rectangle(num_img, point1, point2, color, 1)
        # print(f"boundRect {i} ", boundRect[i])
        # num_cut = num_img_org[point1[1]:point2[1], point1[0]:point2[0]]
        # num_cut_file_path = imgNumNums_file_path[:-4] + '_' + str(i) + ".bmp"
        # cv2.imwrite(num_cut_file_path, num_cut)

    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(imgNumContours_file_path, cv2.hconcat([num_img, binary_img]))
    pass

class CNumRec:
    def __init__(self, template_dir):
        self.m_template_list = []
        for i in range(10):
            template_img = cv2.imread(template_dir + str(i) + ".bmp", cv2.IMREAD_GRAYSCALE)
            self.m_template_list.append(template_img)
        white_template_path = template_dir + "white_template.bmp"
        self.m_img_white_gray = cv2.imread(white_template_path, cv2.IMREAD_GRAYSCALE)

    def findWhiteArea(self, img_gray):
        img_gray_binary = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(img_gray_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        for i, contour in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
        filteredRects = []
        for rect in boundRect:
            if 150000 <= rect[2] * rect[3] < 450000:
                filteredRects.append(rect)
        result_rect = (0, 0, 0, 0)
        if filteredRects:
            result_rect = filteredRects[0]

        drawing = draw_contours(img_gray, contours, hierarchy)
        point1 = (int(result_rect[0]-4), int(result_rect[1])-4)
        point2 = (int(result_rect[0]+result_rect[2]+4), int(result_rect[1]+result_rect[3])+4)
        drawing = cv2.rectangle(drawing, point1, point2, (0, 0, 255), 3)
        cv2.imwrite(imgallContours_file_path, drawing)
        return result_rect

    def findNumArea(self, img_cut_binary_img):
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 3))
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        kernel7 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        img_openning = cv2.morphologyEx(img_cut_binary_img, cv2.MORPH_OPEN, kernel3)
        img_closing = cv2.morphologyEx(img_openning, cv2.MORPH_CLOSE, kernel)
        img_white_binary = cv2.dilate(img_closing, kernel7, iterations=1)
        # Find contours
        contours, hierarchy = cv2.findContours(img_white_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        for i, contour in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(contour, 2, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])
        result_rect = (0, 1000, 0, 0)
        for rect in boundRect:
            if rect[2] / rect[3] >= 4.0 and result_rect[1] > rect[1] and rect[1] > 200:
                result_rect = rect

        drawing = draw_contours(img_cut_binary_img, contours, hierarchy)
        point1 = (int(result_rect[0]-4), int(result_rect[1])-4)
        point2 = (int(result_rect[0]+result_rect[2]+4), int(result_rect[1]+result_rect[3])+4)
        drawing = cv2.rectangle(drawing, point1, point2, (0, 0, 255), 3)
        cv2.imwrite(imgWhiteContours_file_path, drawing)
        return result_rect

    def x_projection(self, binary_img):
        horizontal_projection = np.sum(binary_img, axis=1)
        result = [0,binary_img.shape[0]]
        for i, cnt in enumerate(horizontal_projection):
            if cnt > 0:
                result[0] = i
                break
        for i, cnt in reversed(list(enumerate(horizontal_projection))):
            if cnt > 0:
                result[1] = i
                break
        return result

    def y_projection(self, binary_img):
        binary_img = cv2.threshold(binary_img, 250, 1, cv2.THRESH_BINARY)[1]
        vertical_projection = np.sum(binary_img, axis=0)
        # 找白色区域
        white_area = []
        start = None
        for i, cnt in enumerate(vertical_projection):
            if cnt > 1:
                if start is None:
                    start = i
            elif cnt == 0 and start is not None:
                white_area.append([start, i-1])
                start = None
        white_area_merge = []
        for i in range(len(white_area)-1):
            area = white_area[i]
            if area[1] - area[0] > 12:
                white_area_merge.append(area)
            else:
                area_pre = white_area_merge[-1]
                if area[0] - area_pre[1] <= 6 and area[1] - area_pre[0] < 30:
                    white_area_merge[-1][1] = area[1]
                else:
                    white_area_merge.append(area)
        return white_area_merge


    def getNumBox(self, binary_img):
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_poly = [None] * len(contours)
        boundRect = []
        for i, contour in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(contour, 3, True)
            rect = cv2.boundingRect(contours_poly[i])
            if rect[2] * rect[3] > 200:
                boundRect.append(rect)

        boundRect = sorted(boundRect, key=lambda box: box[0])
        return boundRect

    def processImage(self, img_gray):
        # 查找白色区域
        try:
            whiteArea_rect = self.findWhiteArea(img_gray)

            img_cut = img_gray[whiteArea_rect[1]:whiteArea_rect[1] + whiteArea_rect[3],
                      whiteArea_rect[0]:whiteArea_rect[0] + whiteArea_rect[2]]

            img_cut = cv2.resize(img_cut, (693, 417), interpolation=cv2.INTER_LINEAR)
            img_cut_binary_img = cv2.threshold(img_cut, 240, 255, cv2.THRESH_BINARY_INV)[1]
            # 查找数字区域
            result_rect = self.findNumArea(img_cut_binary_img)
            num_img = img_cut[result_rect[1]:result_rect[1] + result_rect[3],
                      result_rect[0]:result_rect[0] + result_rect[2]]

            binary_img = cv2.threshold(num_img, 190, 255, cv2.THRESH_BINARY_INV)[1]
            # 大小标准化
            # num_img_h, num_img_w = binary_img.shape[0], binary_img.shape[1]
            # ratio = num_img_h / 35
            # num_img_w_new = int(num_img_w / ratio)
            # binary_img = cv2.resize(binary_img, (num_img_w_new, 35), interpolation=cv2.INTER_LINEAR)
            # num_img = cv2.resize(num_img, (num_img_w_new, 35), interpolation=cv2.INTER_LINEAR)

            # y投影
            num_area_x_list = self.y_projection(binary_img)
            num_area_y_list = []
            for num_area_x in num_area_x_list:
                binary_img_oneNum = binary_img[:, num_area_x[0]:num_area_x[1]]
                num_area_y = self.x_projection(binary_img_oneNum)
                num_area_y_list.append(num_area_y)
            # 获取数字box
            imgNumArea(num_img, binary_img)
            result_boxes = self.getNumBox(binary_img)

            # 识别数字
            str_result = ""
            for index in range(7):
                x, y, w, h = result_boxes[index]
                y_new = y - 1
                h_new = h+2
                if y_new < 0:
                    y_new = 0
                if y_new + h_new > num_img.shape[0]:
                    h_new = num_img.shape[0] - y_new
                x_new = x - 2
                if x_new < 0:
                    x_new = 0
                w_new = w + 4
                img_tmp = num_img[y_new:y_new+h_new, x_new:x_new+w_new]
                img_tmp = cv2.resize(img_tmp, self.m_template_list[0].shape[::-1], interpolation=cv2.INTER_LINEAR)
                max_score = 0
                num_result = "_"
                scores = []
                # img_tmp = cv2.threshold(img_tmp, 240, 255, cv2.THRESH_BINARY_INV)[1]
                cv2.imwrite(imgNumNums_dir + imgall_filename + "_" + str(index) + ".jpg", img_tmp)
                for i in range(len(self.m_template_list)):
                    template_img = self.m_template_list[i]
                    # template_img = cv2.threshold(template_img, 240, 255, cv2.THRESH_BINARY_INV)[1]
                    # img_tmp = cv2.threshold(img_tmp, 240, 255, cv2.THRESH_BINARY_INV)[1]
                    # cv2.imwrite("img.jpg", template_img)
                    try:
                        res = cv2.matchTemplate(img_tmp, template_img, cv2.TM_CCOEFF_NORMED)
                    except:
                        pass
                    max_ = res.max()
                    scores.append(max_)
                    if max_ > max_score:
                        max_score = max_
                        num_result = i
                scores_indices = np.argsort(scores)
                if w > 12:
                    num_result = scores_indices[-1]
                    if num_result == 1:
                        num_result = scores_indices[-2]
                else:
                    num_result = 1
                print(index, "scores:", scores)
                str_result = str_result + str(num_result)
            cv2.imwrite(imgNumResult_dir + imgall_filename + str_result + ".bmp", num_img)
            cv2.imwrite(imgNumAera_file_path + imgall_filename + str_result + ".bmp", binary_img)

            return str_result
        except:
            pass
    def processImage1(self, img_gray):
        # 查找白色区域
        whiteArea_rect = self.findWhiteArea(img_gray)

        img_cut = img_gray[whiteArea_rect[1]:whiteArea_rect[1] + whiteArea_rect[3],
                  whiteArea_rect[0]:whiteArea_rect[0] + whiteArea_rect[2]]

        img_cut = cv2.resize(img_cut, (693, 417), interpolation=cv2.INTER_LINEAR)
        img_cut_binary_img = cv2.threshold(img_cut, 250, 255, cv2.THRESH_BINARY_INV)[1]
        # 查找数字区域
        result_rect = self.findNumArea(img_cut_binary_img)
        binary_img = img_cut_binary_img[result_rect[1]:result_rect[1] + result_rect[3],
                     result_rect[0]:result_rect[0] + result_rect[2]]
        num_img = img_cut[result_rect[1]:result_rect[1] + result_rect[3],
                  result_rect[0]:result_rect[0] + result_rect[2]]

        # 获取数字box
        result_boxes = self.getNumBox(binary_img)

        # Apply vertical projection to the binary image to get the sum of white pixels in each column
        binary_img = cv2.threshold(binary_img, 250, 1, cv2.THRESH_BINARY)[1]

        vertical_projection = np.sum(binary_img, axis=0)

        arr = vertical_projection
        arr_size = len(vertical_projection)

        # Now you can access the elements of the array using the [] operator
        white_x_min = []
        white_x_max = []
        step = 10
        run_one_flg = False
        for i in range(arr_size):
            element = arr[i]
            # Do something with the element
            if element == 0:
                if run_one_flg and step > 2:
                    white_x_max.append(i)
                    step = 0
                    run_one_flg = False
                else:
                    step += 1
            elif element > 0:
                if not run_one_flg and step > 2:
                    white_x_min.append(i)
                    step = 0
                    run_one_flg = True
                else:
                    step += 1
        save_num_line(num_img, white_x_min, white_x_max)
        # Recognize digits
        result = []  # class, xmin, ymin
        str_result = ""
        xmin_ = 0
        xmax_ = 0
        for index in range(7):
            xmin_ = white_x_min[index] - 3
            if xmin_ < 0:
                xmin_ = 0

            xmax_ = white_x_max[index]
            if xmax_ + 6 - xmin_ < 25:
                xmax_ = xmin_ + 25 - 6
            img_tmp = num_img[0:num_img.shape[0], xmin_:xmax_ + 6]
            max_score = 0
            loc = None
            xmin__ = 0
            res_loc = []
            for i in range(len(self.m_template_list)):
                template_img = self.m_template_list[i]
                res = cv2.matchTemplate(img_tmp, template_img, cv2.TM_CCOEFF_NORMED)
                max_ = cv2.minMaxLoc(res)[1]
                if max_ > max_score:
                    threshold = max_
                    loc = np.where(res >= threshold)
                    xmin__ = loc[1][0]
                    ymin_ = loc[0][0]
                    max_score = max_
                    max_index = i
                    res_loc = [i, xmin_ + xmin__, ymin_]
            str_result = str_result + str(res_loc[0])
            result.append(res_loc)

        for vec in result:
            for elem in vec:
                print(elem, end=" ")
            print()
        grayscale_image = cv2.cvtColor(num_img, cv2.COLOR_GRAY2BGR)
        for index in range(7):
            xmin_ = white_x_min[index]
            xmax_ = white_x_max[index]
            cv2.line(grayscale_image, (xmin_, 0), (xmin_, num_img.shape[0]), (0, 0, 255), 1)
            cv2.line(grayscale_image, (xmax_, 0), (xmax_, num_img.shape[0]), (0, 0, 255), 1)
        # cv2.namedWindow("Binary image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Binary image", grayscale_image)
        # cv2.moveWindow("Binary image", 100, 100)
        # cv2.waitKey(0)
        return str_result


def NumRecTest(img_path):
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    nr = CNumRec(template_dir)
    result = nr.processImage(img_gray)
    print(result)

def NumRecTestDir():
    global imgall_filename
    global imgallContours_file_path
    global imgWhiteContours_file_path
    global imgNumLine_file_path
    global imgNumContours_file_path
    global imgNumNums_file_path
    for filename in os.listdir(imgall):
        print(">>> ",filename)
        imgall_filename = filename
        imgall_img_path = imgall + f"{imgall_filename}"
        imgallContours_file_path = imgallContours_dir + f"{imgall_filename}"
        imgWhiteContours_file_path = imgWhiteContours_dir + f"{imgall_filename}"
        imgNumLine_file_path = imgNumLine_dir + f"{imgall_filename}"
        imgNumContours_file_path = imgNumContours_dir + f"{imgall_filename}"
        imgNumNums_file_path = imgNumNums_dir + f"{imgall_filename}"
        imgNumAera_file_path = imgNumAera_dir + f"{imgall_filename}"

        img_gray = cv2.imread(imgall_img_path, cv2.IMREAD_GRAYSCALE)
        nr = CNumRec(template_dir)
        str_result = nr.processImage(img_gray)
        print(str_result)

if __name__ == '__main__':
    root_dir = r"D:\02dataset\01work\05nanjingLG\03NumRec/"
    imgall = root_dir + "imgall0422/"
    template_dir = root_dir + "template/"
    imgallContours_dir = root_dir + "imgallContours/"
    imgallWhiteArea = root_dir + "imgallWhiteArea/"
    imgWhiteContours_dir = root_dir + "imgWhiteContours/"
    imgNumContours_dir = root_dir + "imgNumContours/"
    imgNumAera_dir = root_dir + "imgNumAera/"
    imgNumNums_dir = root_dir + "imgNumNums/"
    imgNumLine_dir = root_dir + "imgNumLine/"
    imgNumResult_dir = root_dir + "imgNumResult/"
    imgall_filename = "img_3_0538062.bmp"
    imgall_img_path = imgall + f"{imgall_filename}"
    imgallContours_file_path = imgallContours_dir + f"{imgall_filename}"
    imgWhiteContours_file_path = imgWhiteContours_dir + f"{imgall_filename}"
    imgNumLine_file_path = imgNumLine_dir + f"{imgall_filename}"
    imgNumContours_file_path = imgNumContours_dir + f"{imgall_filename}"
    imgNumNums_file_path = imgNumNums_dir + f"{imgall_filename}"
    imgNumAera_file_path = imgNumAera_dir + f"{imgall_filename}"
    NumRecTest(imgall_img_path)
    try:
        # NumRecTestDir()
        pass
    except:
        pass