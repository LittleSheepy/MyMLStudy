




































import os
import xml.etree.ElementTree as ET

def parse_xml_VisionMaster(xml_filepath):
    tree = ET.parse(xml_filepath)
    root = tree.getroot()
    objects_info = []
    _ItemsData = root.find("_ItemsData")
    for obj in _ItemsData.findall('VisionMaster.ModuleMainWindow.ModuleDialogNew.DeepLearning.FlawPolygonRoiParameter'):
        flags = obj.find('flags').text
        _PolygonPoints = obj.find('_PolygonPoints')
        Points = []
        for PolygonPoint in _PolygonPoints.findall('HikPcUI.ImageView.PolygonPoint'):
            x = str(round(float(PolygonPoint.find('x').text)/2448.0, 4))
            y = str(round(float(PolygonPoint.find('y').text)/2048.0, 4))
            Points.append([x, y])
        objs = {
            "flags": flags,
            "Points": Points
        }
        objects_info.append(objs)
    return objects_info

def VisionMaster2coco128Seg(xml_src, txt_dir):
    for file_name in os.listdir(xml_src):
        xml_filepath = xml_src + file_name
        objects_info = parse_xml_VisionMaster(xml_filepath)
        txt_filepath = txt_dir + file_name[:-3] + "txt"
        with open(txt_filepath, "w") as f:
            for contour in objects_info:
                line = ""
                line = line + "0"
                for point in contour["Points"]:
                    line = line + " " + point[0] + " " + point[1]
                line = line + "\n"
                f.write(line)



if __name__ == '__main__':
    dir_root = r"D:\04DataSets\02/"
    xml_src = dir_root + r"/VisionMaster_xml/"
    txt_dir = dir_root + r"/txt_yolo/"
    xml_filepath = r"D:\04DataSets\02\VisionMasterTemplate.xml"
    # objects_info = parse_xml_VisionMaster(xml_filepath)
    # print(objects_info)

    VisionMaster2coco128Seg(xml_src, txt_dir)


