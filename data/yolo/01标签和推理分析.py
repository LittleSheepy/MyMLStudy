import os, shutil


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

    unmatched_list2 = len(list2)

    if unmatched_list1 == 0 and unmatched_list2 == 0:
        return "OK"
    elif unmatched_list1 > 0:
        return "MISS"
    else:
        return "MORE"
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


def FilterResult():
    for img_file in os.listdir(dir_img_compare):
        label_file = img_file[:-4] + ".txt"
        txt_label_path = dir_txt_label + label_file
        txt_pre_path = dir_txt_pre + label_file
        label_bbox_list = txt2_yolo_bbox_list(txt_label_path)
        pre_bbox_list = txt2_yolo_bbox_list(txt_pre_path)
        result = compare_bbox_lists(label_bbox_list, pre_bbox_list)
        shutil.copyfile(dir_img_compare + img_file, dir_root + "/img_compare" + result + "/" + img_file)



if __name__ == '__main__':
    dir_root = r"D:\0\0LG_DATA\01coco128little0606/"
    dir_txt_label = dir_root + r"\txt_train_have/"
    dir_txt_pre = dir_root + r"\txt_labels/"

    dir_img_compare = dir_root + r"\img_compare/"
    dir_img_compare_OK = dir_root + r"\img_compareOK/"
    dir_img_compare_MISS = dir_root + r"\img_compareMISS/"
    dir_img_compare_MORE = dir_root + r"\img_compareMORE/"
    FilterResult()
    print("\nover")

