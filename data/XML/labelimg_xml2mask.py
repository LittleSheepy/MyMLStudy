import xml.etree.ElementTree as ET
import os
import numpy as np
import cv2

"""类别字典的创建 class_name:序号 """
CLASSES_NAME = (
    "__background__ ",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)
name2id = dict(zip(CLASSES_NAME, range(len(CLASSES_NAME))))


def get_xml_label(label_path):
    """从xml文件中获得label"""

    anno = ET.parse(label_path).getroot()  # .getroot()获取根节点
    # for node in anno:  # 子树
    #     print(node.tag,node.attrib)  # 节点名称以及节点属性（含object物体）

    boxes = []
    classes = []
    for obj in anno.iter("object"):  # 迭代object的子节点
        # for i in obj:
        #     print(i)  # object的子节点含:name pose truncated occluded bndbox difficult

        # 放弃难分辨的图片
        difficult = int(obj.find("difficult").text) == 1
        if difficult:
            continue

        # bounding box坐标值的查找
        _box = obj.find("bndbox")
        box = [
            _box.find("xmin").text,
            _box.find("ymin").text,
            _box.find("xmax").text,
            _box.find("ymax").text,
        ]

        # 框像素点位置-1(python从0开始)
        TO_REMOVE = 1
        box = tuple(
            map(lambda x: x - TO_REMOVE, list(map(float, box)))
        )
        boxes.append(box)

        # 框对应的类别序号
        name = obj.find("name").text.lower().strip()  # 类别名称，统一为小写，并且去除左右空格以及换行符
        classes.append(name2id[name])  # 序号

    boxes = np.array(boxes, dtype=np.float32)
    return boxes, classes


# label_path = os.path.join(r'D:\VOC2012\Annotations', '%s.xml')  # %s指待输入的字符串
# boxes, classes = get_xml_label(label_path % '2008_000007')
# print(boxes)
# print(classes)
def xml2mask(xml_path):
    anno = ET.parse(xml_path).getroot()  # .getroot()获取根节点
    size = anno.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)
    img_mask = np.ones((height, width))
    for obj in anno.iter("object"):
        _box = obj.find("bndbox")
        box = [
            _box.find("xmin").text,
            _box.find("ymin").text,
            _box.find("xmax").text,
            _box.find("ymax").text,
        ]
        box = tuple(
            map(lambda x: x, list(map(int, box)))
        )
        img_mask[box[1]:box[3], box[0]:box[2]] = 0
    return img_mask
def xml2maskByDir(dir_xml, dir_mask):
    for fileName in os.listdir(dir_xml):
        xml_path = dir_xml + fileName
        img_mask = xml2mask(xml_path)
        cv2.imwrite(dir_mask + fileName[:-3]+"png", img_mask*255)
if __name__ == '__main__':
    dir_root = r"D:\04DataSets\02/"
    dir_xml = dir_root + r"labelme_xml/"
    dir_mask = dir_root + r"labelme_mask/"
    xml2maskByDir(dir_xml, dir_mask)