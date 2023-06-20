import os, shutil
import cv2

def readTxt(txt_path, w, h):
    rectList = []
    if os.path.isfile(txt_path):
        with open(txt_path, "r") as f:
            for line in f:
                data = line.split(" ")
                cls = data[0]
                cx = float(data[1])*w
                cy = float(data[2])*h
                dw = float(data[3])*w
                dh = float(data[4])*h
                lx = cx - (dw/2)
                ly = cy - (dh/2)
                rtRect = [lx, ly, dw, dh]
                rectList.append(rtRect)
    return rectList

def drawBox(img, rectList, color=(0, 255, 0)):
    for rect in rectList:
        cv2.rectangle(img, (int(rect[0]), int(rect[1])), (int(rect[0] + rect[2]), int(rect[1] + rect[3])),
                  color, 1)
    return img

def compare():
    for filename in os.listdir(img_dir):
        img_name = filename[:-4]
        img_label = cv2.imread(img_dir+filename)
        img_predict = cv2.imread(img_predict_path + filename)
        img_pre = cv2.imread(img_dir+filename)
        h, w = img_pre.shape[1], img_pre.shape[0]

        txt_label_path = txt_label + img_name + ".txt"
        rectList_label = readTxt(txt_label_path, w, h)
        img_label = drawBox(img_label, rectList_label)

        txt_pre_path = txt_pre + img_name + ".txt"
        rectList_pre = readTxt(txt_pre_path, w, h)
        img_pre = drawBox(img_pre, rectList_pre)

        result_img = cv2.hconcat([img_label, img_pre, img_predict])
        cv2.imwrite(img_save + filename, result_img)


class YoloBbox:
    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

    def is_overlap(self, box):
        return not (self.x1 >= box.x2 or self.x2 <= box.x1 or self.y1 >= box.y2 or self.y2 <= box.y1)

    def iou(self, box):
        if not self.is_overlap(box):
            return 0

        intersection_x1 = max(self.x1, box.x1)
        intersection_y1 = max(self.y1, box.y1)
        intersection_x2 = min(self.x2, box.x2)
        intersection_y2 = min(self.y2, box.y2)

        intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)

        self_area = (self.x2 - self.x1) * (self.y2 - self.y1)
        box_area = (box.x2 - box.x1) * (box.y2 - box.y1)

        union_area = self_area + box_area - intersection_area

        return intersection_area / union_area


def compare_bbox_lists(list1, list2, iou_threshold=0.5):
    matched = 0
    unmatched_list1 = 0
    unmatched_list1_list = []
    unmatched_list2 = 0

    for bbox1 in list1:
        max_iou = 0
        matched_bbox = None
        for bbox2 in list2:
            iou = bbox1.iou(bbox2)
            if iou > max_iou:
                max_iou = iou
                matched_bbox = bbox2

        if max_iou > iou_threshold:
            matched += 1
            list2.remove(matched_bbox)
        else:
            unmatched_list1 += 1
            unmatched_list1_list.append(bbox1)

    unmatched_list2 = len(list2)

    if unmatched_list1 == 0 and unmatched_list2 == 0:
        return "OK", unmatched_list1_list, list2
    elif unmatched_list1 > 0:
        return "MISS", unmatched_list1_list, list2
    else:
        return "MORE", unmatched_list1_list, list2
    #return matched, unmatched_list1, unmatched_list2


def txt2_yolo_bbox_list(txt_path, w=248, h=248):
    defectList = []

    if not os.path.exists(txt_path):
        return []
    with open(txt_path, 'r') as file:
        for line in file:
            if line == "":
                continue
            defect = YoloBbox()
            data = line.split(" ")
            cls = data[0]
            cx = float(data[1]) * w
            cy = float(data[2]) * h
            dw = float(data[3]) * w
            dh = float(data[4]) * h
            defect.x1 = cx - (dw / 2)
            defect.y1 = cy - (dh / 2)
            defect.x2 = cx + (dw / 2)
            defect.y2 = cy + (dh / 2)
            defectList.append(defect)
    return defectList

def drawBoxByYoloBbox(img, rectList, color=(0, 255, 0)):
    for rect in rectList:
        cv2.rectangle(img, (int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)),
                  color, 1)
    return img
def FilterResult():
    for txt_file in os.listdir(dir_txt_have):
        print(txt_file)
        img_file = txt_file[:-4] + ".jpg"
        img_predict = cv2.imread(dir_img_pre + img_file)
        img_label = cv2.imread(dir_img_train+img_file)
        img_pre = cv2.imread(dir_img_train+img_file)
        if img_predict is None:
            continue
        if img_label is None:
            continue
        if img_pre is None:
            continue
        txt_label_path = dir_txt_label + txt_file
        txt_pre_path = dir_txt_pre + txt_file
        label_bbox_list = txt2_yolo_bbox_list(txt_label_path)
        pre_bbox_list = txt2_yolo_bbox_list(txt_pre_path)
        result, unmatched_list1_list, list2 = compare_bbox_lists(label_bbox_list.copy(), pre_bbox_list.copy())
        img_save_path = dir_root + "/img_compare" + result + "/" + img_file
        img_label = drawBoxByYoloBbox(img_label, label_bbox_list, (0, 255, 0))
        img_label = drawBoxByYoloBbox(img_label, unmatched_list1_list, (0, 0, 255))
        img_pre = drawBoxByYoloBbox(img_pre, pre_bbox_list, (0, 255, 0))
        img_pre = drawBoxByYoloBbox(img_pre, list2, (0, 0, 255))
        result_img = cv2.hconcat([img_label, img_pre, img_predict])
        cv2.imwrite(img_save_path, result_img)

        #shutil.copyfile(dir_img_compare + img_file, img_save_path)

def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

if __name__ == '__main__':
    dir_root = r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict29\/"
    img_dir = dir_root + r"img_have/"
    img_predict_path = dir_root + r"img_pre/"
    img_save = dir_root + r"img_duibi/"
    txt_label = dir_root + r"txt_train/"
    txt_pre = dir_root + r"txt_labels/"
    # compare()

    dir_root = r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict29\/"
    dir_img_train = dir_root + r"\img_train/"
    dir_img_pre = dir_root + r"\img_pre/"
    dir_txt_have = dir_root + r"\txt_have/"
    dir_txt_label = dir_root + r"\txt_train/"
    dir_txt_pre = dir_root + r"\txt_labels/"

    dir_img_compare = dir_root + r"\img_compare/"

    dir_img_compare_OK = dir_root + r"\img_compareOK/"
    dir_img_compare_MISS = dir_root + r"\img_compareMISS/"
    dir_img_compare_MORE = dir_root + r"\img_compareMORE/"
    mk_dir(dir_img_compare)
    mk_dir(dir_img_compare_OK)
    mk_dir(dir_img_compare_MISS)
    mk_dir(dir_img_compare_MORE)
    FilterResult()
    print("\nover")

