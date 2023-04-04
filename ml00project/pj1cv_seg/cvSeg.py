import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import psutil
import gc
from memory_profiler import profile

def Blur(image):
    # 高斯去噪
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image

def GetHSV(img_bgr):
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_h = img_hsv[..., 0]
    img_s = img_hsv[..., 1]
    img_v = img_hsv[..., 2]
    return img_h, img_s, img_v
def plt_show(title, image):
    plt.title(title)
    #plt.plot(image, cmap='gray')
    plt.imshow(image, cmap='gray')
    plt.show()
def BlurGray(img_gray):
    img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for i in range(10):
        img_bgr = Blur(img_bgr)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    #img_h, img_s, img_v = GetHSV(img_bgr)
    return img_gray
# 开运算
def OpeningOperation(img_gray):
    # 先腐蚀后膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))      # 结构元素，矩形大小3*3
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)          # 支持形态学的函数，此参数为开操作，先腐蚀后膨胀，会消除一些为1的白色噪点
    return img_gray

def Gray2Mask(img_gray, thresh=None):
    img_blur = BlurGray(img_gray)
    plt_show("img_h_blur", img_blur)
    # Ostu阈值分割
    mean = thresh or (np.mean(img_blur)+30)
    ret, img_th1 = cv2.threshold(img_blur, mean, 255, cv2.THRESH_BINARY)
    plt_show("img_th1", img_th1)
    # 先腐蚀后膨胀 开运算
    img_open = OpeningOperation(img_th1)
    plt_show("img_open", img_open)
    return img_open

def py_nms(dets, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的
    order = scores.argsort()[::-1]
    # print("order:",order)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

def matchTemplate_max(img_gray, template, method=cv2.TM_CCOEFF_NORMED):
    res = cv2.matchTemplate(img_gray, template, method)
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
    #res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    threshold_max = np.max(res)
    #print(threshold_max)
    threshold = threshold_max
    loc = np.where(res >= threshold)
    score = res[res >= threshold]  # 大于模板阈值的目标置信度
    return loc, score

def matchTemplate_min(img_gray, template):
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # res = cv2.matchTemplate(img_gray, template, cv2.TM_CCORR_NORMED)
    res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
    threshold_min = np.min(res)
    #print(threshold_max)
    threshold = threshold_min
    loc = np.where(res <= threshold)
    score = res[res <= threshold]  # 大于模板阈值的目标置信度
    return loc, score

def matchTemplate(img_gray, template, template_threshold=0.5, method=cv2.TM_CCOEFF_NORMED):
    w, h = template.shape[::-1]
    loc, _ = matchTemplate_max(img_gray, template)
    pt = [loc[0][0],loc[1][0]]
    template_new = img_gray[pt[0]:pt[0]+h, pt[1]:pt[1]+w]
    # plt.imshow(template_new, cmap="gray")
    # plt.show()
    res = cv2.matchTemplate(img_gray, template_new, method)
    threshold_max = np.max(res)
    #print(threshold_max)
    threshold = template_threshold
    loc = np.where(res >= threshold)
    score = res[res >= threshold]  # 大于模板阈值的目标置信度
    return loc, score


def template_nms(img_gray, template_img, template_threshold):
    '''
    img_gray:待检测的灰度图片格式
    template_img:模板小图，也是灰度化了
    template_threshold:模板匹配的置信度
    '''

    h, w = template_img.shape[:2]
    # res = cv2.matchTemplate(img_gray, template_img, cv2.TM_CCOEFF_NORMED)
    # #start_time = time.time()
    # loc = np.where(res >= template_threshold)  # 大于模板阈值的目标坐标
    # score = res[res >= template_threshold]  # 大于模板阈值的目标置信度
    loc, score = matchTemplate(img_gray, template_img, template_threshold)
    # 将模板数据坐标进行处理成左上角、右下角的格式
    xmin = np.array(loc[1])
    ymin = np.array(loc[0])
    xmax = xmin + w
    ymax = ymin + h
    xmin = xmin.reshape(-1, 1)  # 变成n行1列维度
    xmax = xmax.reshape(-1, 1)  # 变成n行1列维度
    ymax = ymax.reshape(-1, 1)  # 变成n行1列维度
    ymin = ymin.reshape(-1, 1)  # 变成n行1列维度
    score = score.reshape(-1, 1)  # 变成n行1列维度
    data_hlist = []
    data_hlist.append(xmin)
    data_hlist.append(ymin)
    data_hlist.append(xmax)
    data_hlist.append(ymax)
    data_hlist.append(score)
    data_hstack = np.hstack(data_hlist)  # 将xmin、ymin、xmax、yamx、scores按照列进行拼接

    thresh = 0.1  # NMS里面的IOU交互比阈值
    keep_dets = py_nms(data_hstack, thresh)
    #print("nms time:", time.time() - start_time)  # 打印数据处理到nms运行时间
    dets = data_hstack[keep_dets]  # 最终的nms获得的矩形框
    return dets
def save_det(img_rgb, dets):
    for coord in dets:
        cv2.rectangle(img_rgb, (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3])), (0, 0, 255), 2)
    cv2.imwrite("save_det_result.jpg", img_rgb)
def imgdel(img_test, template):
    plt_show("img_test", img_test)
    plt_show("template", template)
    img_del1 = cv2.subtract(img_test, template)
    img_del2 = cv2.subtract(template, img_test)
    img_del = img_del1 + img_del2
    #img_del = abs(img_test - template)
    plt_show("img_del", img_del)
    img_test = BlurGray(img_test)
    template = BlurGray(template)
    plt_show("img_testBlur", img_test)
    plt_show("templateBlur", template)
    #img_del = abs(img_test - template)
    img_del1 = cv2.subtract(img_test, template)
    img_del2 = cv2.subtract(template, img_test)
    img_del = img_del1 + img_del2
    #ret, img_th1 = cv2.threshold(img_del, 50, 0, cv2.THRESH_BINARY_INV)
    ret, img_th1 = cv2.threshold(img_del, 50, 0, cv2.THRESH_TOZERO)
    plt_show("img_delBlur", img_th1)
    pass

def save_max_template(dir_images, template_path, dir_images_max_template):
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template_gray.shape[:2]
    for imageName in os.listdir(dir_images):
        img_bgr = cv2.imread(dir_images + imageName)
        img_gray = cv2.imread(dir_images + imageName, cv2.IMREAD_GRAYSCALE)
        loc, score = matchTemplate_max(img_gray, template_gray)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
            cv2.putText(img_bgr, str(score[0]), pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
        cv2.imwrite(dir_images_max_template + imageName, img_bgr)

def resultIni(ptxy=[], locInRes=[], scoreMap={}, flg="unkonw", w=78, h=120):
    result = {}
    result["ptxy"] = ptxy
    result["w"] = w
    result["h"] = h
    result["flg"] = flg
    result["scoreMap"] = scoreMap
    result["locInRes"] = locInRes
    return result
def cutImage(img, ptxy, wh, extend):
    img_h, img_w = img.shape[0], img.shape[1]
    pt_left = ptxy[0]-extend
    if pt_left < 0:
        pt_left = 0
    pt_top = ptxy[1]-extend
    if pt_top < 0:
        pt_top = 0
    pt_right = ptxy[0]+wh[0]+extend
    if pt_right > img_w:
        pt_right = img_w
    pt_bottom = ptxy[1]+wh[1]+extend
    if pt_bottom > img_h:
        pt_bottom = img_h
    return img[pt_top:pt_bottom, pt_left:pt_right]

def matchDefect(img_gray, defectDict):
    matchDefectResult = {}
    matchDefectResult["score"] = 0
    for defect2Name, defect2_gray in defectDict.items():
        loc, score = matchTemplate_max(img_gray, defect2_gray)
        if score[0] > matchDefectResult["score"]:
            matchDefectResult["name"] = defect2Name
            matchDefectResult["loc"] = loc
            matchDefectResult["score"] = score[0]
    return matchDefectResult

def getmax(scoreMap):
    score_max = 0
    for ix in scoreMap:
        for iy in scoreMap[ix]:
            if scoreMap[ix][iy] > score_max:
                score_max = scoreMap[ix][iy]
    return score_max


def wrongProcess(result, flg, by, scoreDefect, result_cut_extend):
    global wrongname
    PanDuanIf[by] = PanDuanIf.get(by, 0) + 1
    result["flg"] = flg
    result["by"] = by
    wrongscore.append(scoreDefect)
    if not os.path.exists(r"D:\04DataSets\02\wrong/" + by):
        os.mkdir(r"D:\04DataSets\02\wrong/" + by)
    plt.imsave(
        r"D:\04DataSets\02\wrong/" + by+ "/" + global_imagename[-7:-4] + "_" + str(wrongname) + "_" + str(
            scoreDefect) + ".jpg", result_cut_extend)
    wrongname = 1 + wrongname
    # plt.imshow(result_cut_extend)
    # plt.show()
# 递归标记
#@profile
def flg2(img_gray, result, space, resultMap):
    global wrongname
    global defectDict
    global img_global_rbg
    global mask_global
    global global_imagename
    global f_global
    # 查找九宫格
    img_w, img_h = img_gray.shape[::-1]
    pt = result["ptxy"]
    locInRes = result["locInRes"]
    if locInRes == [-1,0]:
        a=1
        pass
    print("遍历：", locInRes)
    w = result["w"]
    h = result["h"]
    space_h, space_w = space
    result_cut = img_gray[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
    result_cut_mask = mask_global[pt[1]:pt[1]+h, pt[0]:pt[0]+w]
    result_cut_mask_sum = np.sum(result_cut_mask)
    result_cut_extend = cutImage(img_gray, pt, [w, h], 10)
    result_cut_extend_rbg = cutImage(img_global_rbg, pt, [w, h], 10)
    wrongFlg = 0
    if result_cut_mask_sum > 2000:
        wrongFlg = 1
    elif result_cut_mask_sum > 1:
        wrongFlg = 2
    if result_cut_mask_sum > 2000:
        mask_name = ""
        mask_name = mask_name + r"D:\04DataSets\02\mask_little/" + "big/" + global_imagename[-7:-4]
        mask_name_rbg = mask_name + "_" + str(result_cut_mask_sum) + ".jpg"
        mask_name_mask = mask_name + "_" + str(result_cut_mask_sum) + ".jpg"
        cv2.imwrite(mask_name_mask, result_cut_mask*255)
        cv2.imwrite(mask_name_rbg, result_cut_extend_rbg)
    elif result_cut_mask_sum > 1:
        mask_name = ""
        mask_name = mask_name + r"D:\04DataSets\02\mask_little/" + "small/" + global_imagename[-7:-4]
        mask_name_rbg = mask_name + "_" + str(result_cut_mask_sum) + ".jpg"
        mask_name_mask = mask_name + "_" + str(result_cut_mask_sum) + ".jpg"
        cv2.imwrite(mask_name_mask, result_cut_mask*255)
        cv2.imwrite(mask_name_rbg, result_cut_extend_rbg)
    scoreMap = result.get("scoreMap", {})
    if (not run_findAndFlg2ByDir == 1) and (showflg == 1):
        img_gray_show = img_gray.copy()
        cv2.putText(img_gray_show, str(locInRes), pt, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
        plt.imshow(img_gray_show)
        ax = plt.gca()
        ax.add_patch(plt.Rectangle(pt, result["w"], result["h"], color="blue", fill=False, linewidth=1))
    for ix in range(-1,2,1):
        for iy in range(-1,2,1):
            # 自己 跳过
            if ix == 0 and iy == 0:
                continue
            #print(ix, iy)
            # 匹配过 跳过
            scoreMap[str(ix)] = scoreMap.get(str(ix), {})
            #scoreMap[str(ix)][str(iy)] = scoreMap.get(str(ix), 0)
            if str(iy) in scoreMap[str(ix)].keys():
                print("匹配过 跳过")
                continue

            # 获取位置
            pt_new = [pt[0]+ix*space_w, pt[1]+iy*space_h]
            # 超过边缘就 continue
            if pt_new[0] < 0 or pt_new[1] < 0 or \
                pt_new[0] + result["w"] > img_w or \
                pt_new[1] + result["h"] > img_h:
                print("超过边缘就 continue")
                continue
            new_cut_mask = labelme_mask_gray_global[pt_new[1]:pt_new[1] + h, pt_new[0]:pt_new[0] + w]
            if np.sum(new_cut_mask) > 0:
                continue
            # 截取
            extend_pix = 20
            cut_w = result["w"] + extend_pix * 2
            cut_h = result["h"] + extend_pix * 2
            pt_extend_lt = [pt_new[0]-extend_pix, pt_new[1]-extend_pix]
            pt_extend_lt = np.array(pt_extend_lt)
            pt_extend_lt = np.where(pt_extend_lt > 0, pt_extend_lt, 0)
            pt_extend_rb = [pt_extend_lt[0]+cut_w, pt_extend_lt[1]+cut_h]
            if pt_extend_rb[0] > img_w:
                pt_extend_rb[0] = img_w
            if pt_extend_rb[1] > img_h:
                pt_extend_rb[1] = img_h
            img_cut = img_gray[pt_extend_lt[1]:pt_extend_rb[1],pt_extend_lt[0]:pt_extend_rb[0]]
            # 最大匹配
            loc, score = matchTemplate_max(img_cut, result_cut)
            scoreMap[str(ix)] = scoreMap.get(str(ix), {})
            scoreMap[str(ix)][str(iy)] = score[0]
            pt_new = [loc[1][0]+pt_extend_lt[0], loc[0][0]+pt_extend_lt[1]]
            # 设置分数
            locInRes_new = [locInRes[0]+ix, locInRes[1]+iy]
            resultMap[str(locInRes_new[0])] = resultMap.get(str(locInRes_new[0]), {})
            resultMap[str(locInRes_new[0])][str(locInRes_new[1])] = resultMap[str(locInRes_new[0])].get(str(locInRes_new[1]), {})
            result_new = resultMap[str(locInRes_new[0])][str(locInRes_new[1])]
            if result_new == {}:
                scoreMap_new = {}
                scoreMap_new[str(0-ix)] = scoreMap_new.get(str(0-ix), {})
                scoreMap_new[str(0-ix)][str(0-iy)] = score[0]
                flg = "unkonw"
                if score[0] < 0.5:
                    matchDefectResult = matchDefect(img_cut, defectDict)
                    locDefect, scoreDefect = matchDefectResult["loc"], matchDefectResult["score"]
                    # defect2_gray = cv2.imread("./defect2.png", cv2.IMREAD_GRAYSCALE)
                    # defect2Left_gray = cv2.imread("./defect2Left.png", cv2.IMREAD_GRAYSCALE)
                    # loc, score = matchTemplate_max(img_cut, defect2_gray)
                    # locLeft, scoreLeft = matchTemplate_max(img_cut, defect2Left_gray)
                    print("defect2_gray score[0] < 0.5 ", scoreDefect)
                    pt_new = [locDefect[1][0]+pt_extend_lt[0], locDefect[0][0]+pt_extend_lt[1]]
                    if scoreDefect < 0.2:
                        flg = "BothNot"
                # 分数大的直接标right
                # if score[0] > 0.7:
                #     flg = "right"
                result_new = resultIni(pt_new, locInRes_new, scoreMap_new, flg=flg)
                resultMap[str(locInRes_new[0])][str(locInRes_new[1])] = result_new
            else:
                scoreMap_new = result_new["scoreMap"]
                scoreMap_new[str(0 - ix)] = scoreMap_new.get(str(0 - ix), {})
                # 没有就插入分数
                if str(0-iy) not in scoreMap_new[str(0-ix)].keys():
                    scoreMap_new[str(0-ix)][str(0-iy)] = score[0]
                    result_new["scoreMap"] = scoreMap_new
            if (not run_findAndFlg2ByDir == 1) and (showflg == 1):
                # 显示
                ax.add_patch(plt.Rectangle(pt_extend_lt, cut_w, cut_h, color="blue", fill=False, linewidth=1))
                ax.add_patch(plt.Rectangle(pt_new, result["w"], result["h"], color="red", fill=False, linewidth=1))
    if (not run_findAndFlg2ByDir == 1) and (showflg == 1):
        plt.show()

    # 判断自己是否right
    if result["flg"] == "unkonw":
        scoreList = []
        scoreAllList = []
        scoreMaxUnkonwList = []
        scoreMaxUnkonwlocList = []
        scoreThereMaxRightList = []
        for ix in range(-1, 2, 1):
            for iy in range(-1, 2, 1):
                if ix == 0 and iy == 0:
                    continue
                locInRes_tmp = [locInRes[0]+ix, locInRes[1]+iy]
                result_tmp0 = resultMap.get(str(locInRes_tmp[0]), {})
                result_tmp = result_tmp0.get(str(locInRes_tmp[1]), {})
                if result_tmp:
                    scoreThereMax = getmax(result_tmp["scoreMap"])
                    scoreAllList.append(scoreMap[str(ix)][str(iy)])
                    if result_tmp["flg"] == "right":
                        try:
                            scoreList.append(scoreMap[str(ix)][str(iy)])
                            scoreThereMaxRightList.append(scoreThereMax)
                        except Exception as e:
                            print(repr(e))
                            pass
                    elif result_tmp["flg"] == "unkonw":
                        # 距离原点近的
                        dis_result_tmp = abs(result_tmp["locInRes"][0])+abs(result_tmp["locInRes"][1])
                        die_self = abs(locInRes[0])+abs(locInRes[1])
                        if dis_result_tmp < die_self:
                            scoreMaxUnkonwList.append(scoreThereMax)
                            scoreMaxUnkonwlocList.append(result_tmp["locInRes"])

        if scoreList:
            scoreRight_mean = np.mean(scoreList)
            scoreRight_max = np.max(scoreList)
            scoreAll_max = np.max(scoreAllList)
            scoreThereMaxRight_max = np.max(scoreThereMaxRightList)   # 周围的
            if scoreMaxUnkonwList:
                scoreMaxUnkonw_max = np.max(scoreMaxUnkonwList)
                if scoreMaxUnkonw_max > scoreAll_max + 0.1:
                    return

            if 1 == 1:
                matchDefectResult = matchDefect(result_cut_extend, defectDict)
                locDefect, scoreDefect = matchDefectResult["loc"], matchDefectResult["score"]
                score_max_dis = scoreRight_max - scoreDefect
                scoreMaxRightSTD = np.std(scoreThereMaxRightList)
                scoreMaxRightSTD_self = np.std(scoreThereMaxRightList + [scoreRight_max])
                there_self_max_dis = scoreThereMaxRight_max - scoreRight_max

            if scoreRight_mean > 0.8:
                PanDuanIf["score_mean08"] = PanDuanIf.get("score_mean08", 0) + 1
                result["flg"] = "right"
                result["by"] = "score_mean08"
            elif scoreRight_max > 0.8:
                PanDuanIf["score_max08"] = PanDuanIf.get("score_max08", 0) + 1
                result["flg"] = "right"
                result["by"] = "score_max08"
            else:
                # 与defet比较
                # plt.show()
                # matchDefectResult = matchDefect(result_cut_extend, defectDict)
                # locDefect, scoreDefect = matchDefectResult["loc"], matchDefectResult["score"]

                defect2c_gray = cv2.imread("./defect2c.png", cv2.IMREAD_GRAYSCALE)
                loc_dc, score_dc = matchTemplate_max(result_cut_extend, defect2c_gray)
                template2c_gray = cv2.imread("./template2c.png", cv2.IMREAD_GRAYSCALE)
                loc_tc, score_tc = matchTemplate_max(result_cut_extend, template2c_gray)
                # plt.imshow(defect2_gray)
                # plt.show()
                score_max_dis = scoreRight_max - scoreDefect
                there_self_max_dis = scoreThereMaxRight_max - scoreRight_max
                w1 = 1
                while w1 > 0:
                    w1 = w1 - 1
                    # 判断错误
                    if scoreDefect > 0.7:
                        # 自己都wrong了 就return吧 浪费时间
                        wrongProcess(result, "wrong", "scoreDefect07", scoreDefect, result_cut_extend_rbg)
                        break
                    # 判断是正确的
                    if len(scoreThereMaxRightList) > 3:
                        scoreMaxRightSTD = np.std(scoreThereMaxRightList)
                        scoreMaxRightSTD_self = np.std(scoreThereMaxRightList + [scoreRight_max])
                        if scoreMaxRightSTD_self/scoreMaxRightSTD < 1.5:
                            PanDuanIf["stds_std<1.5"] = PanDuanIf.get("stds_std<1.5", 0) + 1
                            result["flg"] = "right"
                            result["by"] = "stds_std<1.5"
                            break
                        if scoreMaxRightSTD_self/scoreMaxRightSTD > 2 and scoreDefect > 0.5:
                            PanDuanIf["stds_std>2"] = PanDuanIf.get("stds_std>2", 0) + 1
                            result["flg"] = "wrong"
                            result["by"] = "stds_std>2"
                            break
                        if there_self_max_dis > 0.15:
                            wrongProcess(result, "wrong", "scoreThereMax", scoreDefect, result_cut_extend_rbg)
                            break
                    if (scoreRight_mean > 0.7) and (scoreDefect < 0.7) and (score_max_dis > 0.05) :
                        PanDuanIf["mean07"] = PanDuanIf.get("mean07", 0) + 1
                        result["flg"] = "right"
                        result["by"] = "mean07"
                    elif (scoreRight_mean > 0.6) and (scoreDefect < 0.6) and (score_max_dis > 0.1) and (score_tc[0] > scoreDefect):
                        PanDuanIf["mean06"] = PanDuanIf.get("mean06", 0) + 1
                        result["flg"] = "right"
                        result["by"] = "mean06"
                    elif (scoreRight_max > 0.7) and (scoreDefect < 0.7) and (score_max_dis > 0.2) :
                        PanDuanIf["score_max_dis02"] = PanDuanIf.get("score_max_dis02", 0) + 1
                        result["flg"] = "right"
                    elif scoreDefect > 0.4:
                        wrongProcess(result, "wrong", "scoreDefect04", scoreDefect, result_cut_extend_rbg)
                        break
                    elif scoreDefect > 0.3 and len(scoreList) > 2:
                        wrongProcess(result, "wrong", "scoreDefect03", scoreDefect, result_cut_extend_rbg)
                        break
                    else:
                        result["flg"] = "BothNot"
                        # plt.imshow(result_cut_extend)
                        # plt.show()
                        break
            # 保存依据
            if run_findAndFlg2ByDir:
                write_line = ""
                write_line = write_line + global_imagename + ","
                write_line = write_line + "\"" + str(locInRes) + "\"" + ","
                write_line = write_line + str(scoreRight_mean) + ","
                write_line = write_line + str(scoreRight_max) + ","
                write_line = write_line + str(scoreThereMaxRight_max) + ","
                write_line = write_line + str(there_self_max_dis) + ","
                write_line = write_line + str(scoreMaxRightSTD_self) + ","
                write_line = write_line + str(scoreMaxRightSTD) + ","
                write_line = write_line + str(scoreMaxRightSTD_self/scoreMaxRightSTD) + ","
                write_line = write_line + str(scoreDefect) + ","
                write_line = write_line + str(score_max_dis) + ","

                write_line = write_line + str(wrongFlg)
                write_line = write_line + "\n"
                f_global.write(write_line)
        else:
            # 确定不了就不递归了
            print(">>>>>>看这里>>>>>>>", result)
            return
    if wrongFlg == 1 and result["flg"] == "right":
        wrong_pre_img.append(global_imagename)
        print(">>>>>>>>>global_imagename=", global_imagename)

    if result["flg"] == "wrong":
        return
    # 字典转一维数组
    scoreDict = {}

    for ix in range(-1, 2, 1):
        for iy in range(-1, 2, 1):
            scoreMap_ix = scoreMap.get(str(ix), {})
            score = scoreMap_ix.get(str(iy), 0)
            key = str(ix) + "_" + str(iy)
            scoreDict[key] = score
    scoreDict_sort_list = sorted(scoreDict.items(), key=lambda x: x[1], reverse=True)
    for scoreDict in scoreDict_sort_list:
        if scoreDict[1] <= 0:
            continue
        ix_s, iy_s = scoreDict[0].split("_")
        ix = int(ix_s)
        iy = int(iy_s)
        locInRes_tmp = [locInRes[0]+ix, locInRes[1]+iy]
        result_tmp0 = resultMap.get(str(locInRes_tmp[0]), {})
        result_tmp = result_tmp0.get(str(locInRes_tmp[1]), {})
        if result_tmp and result_tmp["flg"] == "unkonw":
            try:
                flg2(img_gray, result_tmp, space, resultMap)
            except Exception as e:
                print(repr(e))
                pass

    # 递归遍历
    # for ix in range(-1, 2, 1):
    #     for iy in range(-1, 2, 1):
    #         if ix == 0 and iy == 0:
    #             continue
    #         locInRes_tmp = [locInRes[0]+ix, locInRes[1]+iy]
    #         result_tmp0 = resultMap.get(str(locInRes_tmp[0]), {})
    #         result_tmp = result_tmp0.get(str(locInRes_tmp[1]), {})
    #         if result_tmp and result_tmp["flg"] == "unkonw":
    #             try:
    #                 flg2(img_gray, result_tmp, space, resultMap)
    #             except Exception as e:
    #                 print(repr(e))
    #                 pass

# 查找并标记
def findAndFlg2(img_gray, template_gray, space):
    global f_global
    resultMap = {}
    h, w = template_gray.shape[:2]
    loc, score = matchTemplate_max(img_gray, template_gray)
    pt00 = [loc[1][0], loc[0][0]]
    result = {}
    result["ptxy"] = pt00
    result["w"] = w
    result["h"] = h
    result["flg"] = "right"
    result["scoreMap"] = {}
    result["locInRes"] = [0,0]
    resultMap["0"] = resultMap.get("0", {})
    resultMap["0"]["0"] = result
    flg2(img_gray, result, space, resultMap)
    #while True:
    return resultMap

def saveflg(img_bgr, resultMap, savePath):
    #img_bgr = img_bgr.copy()
    color_dict = {"right": (0, 255, 0), "wrong": (0, 0, 255), "BothNot": (255, 0, 0), "unkonw": (255, 255, 255)}

    if run_findAndFlg2ByDir == 1:
        imgnum = savePath[-7:-4]
        cv2.putText(img_bgr, str(img_wrong_num[int(imgnum)-1]), (30,40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    for ix in resultMap:
        for iy in resultMap[ix]:
            result = resultMap[ix][iy]
            if not result:
                continue
            try:
                cv2.rectangle(img_bgr, result["ptxy"], (result["ptxy"][0] + result["w"], result["ptxy"][1] + result["h"]),
                          color_dict[result["flg"]], 1)
            except Exception as e:
                print(repr(e))
                pass
            cv2.putText(img_bgr, result.get("by", ""), result["ptxy"], cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 1)
    cv2.imwrite(savePath, img_bgr)

def findAndFlg2ByDir(dir_images, template_path, dir_images_flg2, space):
    global global_imagename
    global img_global_rbg
    global mask_global
    global f_global
    global labelme_mask_gray_global
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    h, w = template_gray.shape[:2]
    filePath = os.path.join(r"D:\04DataSets\02/", "result.csv")
    if os.path.exists(filePath):
        os.remove(filePath)
    f_global = open(filePath, "w")
    name_line = ""
    name_line = name_line + "file,loc,"
    name_line = name_line + "scoreRight_mean,scoreRight_max,scoreThereMaxRight_max,there_self_max_dis,"
    name_line = name_line + "scoreMaxRightSTD_self,scoreMaxRightSTD,std/std,scoreDefect,score_max_dis,class"
    name_line = name_line + "\n"
    f_global.write(name_line)
    for imageName in os.listdir(dir_images):
        if not os.path.exists(dir_images+ "../mask/" + imageName[:-4] + "_t" + ".bmp"):
            continue
        global_imagename = imageName
        img_bgr = cv2.imread(dir_images + imageName)
        img_global_rbg = img_bgr
        img_gray = cv2.imread(dir_images + imageName, cv2.IMREAD_GRAYSCALE)
        labelme_mask_gray_global = cv2.imread(dir_labelme_mask + imageName, cv2.IMREAD_GRAYSCALE)
        # mask
        mask_global = cv2.imread(dir_images+ "../mask/" + imageName[:-4] + "_t" + ".bmp", cv2.IMREAD_GRAYSCALE)
        mask_global[mask_global == 0] = 1
        mask_global[mask_global == 255] = 0
        resultMap = findAndFlg2(img_gray, template_gray, space)
        try:
            saveflg(img_bgr, resultMap, dir_images_flg2+imageName)
        except Exception as e:
            print(repr(e))
            pass
    f_global.close()

img_wrong_num = [
    0, 0, 1, 0, 0, 1, 2, 1, 2, 0,
    1, 0, 1, 1, 1, 1, 1, 0, 1, 2,
    0, 2, 1, 1, 0, 1, 0, 0, 0, 1,
    1, 1, 1, 1, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 0, 1,
    0, 0, 1, 0, 1, 1, 1, 0, 0, 1,
    1, 0, 1, 1, 1, 0, 1, 0, 1, 1,
    0, 1, 1, 0, 1, 1, 1, 1, 1, 0,
    1, 1, 0, 1, 1, 0, 0, 1, 1, 0,
    1, 0, 1, 0, 1, 0, 1, 1, 0, 0,
    1, 0, 1, 1, 1, 0, 1, 0, 1, 1,
    0, 1, 0, 0, 1, 0, 0, 0, 0, 1,
    1, 0, 1, 0, 1, 0, 1, 1, 1, 0,
    1, 0, 1, 1, 0, 2, 1, 1, 0, 2
]
wrongscore = []
wrong_defectscore = []
wrongname = 0
PanDuanIf = {}
global_imagename = ""
run_findAndFlg2ByDir = 1
showflg = 1
img_global_rbg = []
f_global = None
mask_global = None
wrong_pre_img = []
labelme_mask_gray_global = None
if __name__ == '__main__':
    imgName = "CB_00011.png"
    global_imagename = imgName
    dir_root = r"D:\04DataSets\02/"
    dir_images = dir_root + r"\images/"
    dir_mask = dir_root + r"\mask/"
    dir_labelme_mask = dir_root + r"\labelme_mask/"
    dir_images_max_template = dir_root + r"\images_max_template_sqdiff/"
    dir_images_flg2 = dir_root + r"\images_flg26/"
    img_path = dir_images + imgName
    mask_path = dir_mask + imgName[:-4] + "_t" + ".bmp"
    template_path = "./template.png"
    template2_path = "./template2.png"
    template2CenterBlack_path = "./template2CenterBlack.png"
    template3_path = "./template3.png"
    defect2_path = "./defect2.png"
    defect2Left_path = "./defect2Left.png"
    img_bgr = cv2.imread(img_path)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask_global = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    template2_gray = cv2.imread(template2_path, cv2.IMREAD_GRAYSCALE)
    defect2_gray = cv2.imread(defect2_path, cv2.IMREAD_GRAYSCALE)
    defect2Left_gray = cv2.imread(defect2Left_path, cv2.IMREAD_GRAYSCALE)
    labelme_mask_gray_global = cv2.imread(dir_labelme_mask + imgName, cv2.IMREAD_GRAYSCALE)
    defectDict = {
        "left": defect2Left_gray,
        "common": defect2_gray
    }
    templateDict = {
        "CenterBlack": template2CenterBlack_path,
        "common": template2_gray
    }
    matchDefect(img_gray, defectDict)
    # 保存模版最大位置
    #save_max_template(dir_images, template2_path, dir_images_max_template)
    # 查找并标记
    space = template_gray.shape[:2]
    # img_bgr = Blur(img_bgr)
    # img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # template2_gray = Blur(template2_gray)


    if run_findAndFlg2ByDir == 1:
        findAndFlg2ByDir(dir_images, template2_path, dir_images_flg2, space)
    else:
        img_global_rbg = img_bgr
        mask_global[mask_global == 0] = 1
        mask_global[mask_global == 255] = 0
        resultMap = findAndFlg2(img_gray, template2_gray, template_gray.shape[:2])
        try:
            saveflg(img_bgr, resultMap, dir_root + "findAndFlg2.jpg")
        except Exception as e:
            print(repr(e))
            pass
    print(wrongscore)
    print('A：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
    gc.collect()
    print('B：%.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

    imgdel(defect2_gray, template_gray)

    loc = matchTemplate(defect2_gray, template_gray)
    det = template_nms(img_gray, template_gray, 0.5)
    det = sorted(det, key=lambda x: (x[1], x[0]))
    save_det(img_bgr, det)

    # 显示
    w, h = template_gray.shape[::-1]
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    cv2.imwrite('res.png', img_bgr)
    # cv2.imshow('res.png', img_bgr)
    # cv2.waitKey(0)



    #plt_show("img_bgr", img_bgr)
    # 去噪
    # img_bgr = Blur(img_bgr)
    # img_bgr = Blur(img_bgr)
    # 获得hsv
    img_h, img_s, img_v = GetHSV(img_bgr)

    img_mask = Gray2Mask(img_v, 230)
    # img_h = 255 - img_h
    # img_mask = Gray2Mask(img_h)
    '''轮廓检测与绘制'''
    # 检测轮廓(外轮廓)
    # th1 = cv2.dilate(img_gray_result, None)  # 膨胀，保证同一个字符只有一个外轮廓
    contours, hierarchy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 轮廓可视化
    #th1_bgr = cv2.cvtColor(th1, cv2.COLOR_GRAY2BGR)  # 转为三通道图

    cv2.drawContours(img_bgr, contours, -1, (0, 0, 255), 2)  # 轮廓可视化

    cv2.imwrite("img_bgr_Contours_img_v.jpg", img_bgr)

    #cv2.waitKey()