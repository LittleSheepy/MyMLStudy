import os
import cv2
import xml.etree.ElementTree as ET

# Define the path to the VOC dataset
voc_path = "/path/to/voc/dataset"
def getRowWhitePoint(img_gray, point_y=500):
    w = img_gray.shape[1]
    RowPoints = {}
    for point_x in range(w):
        if img_gray[point_y, point_x] > 10:
            RowPoints["whiteleft"] = [point_x, point_y]
            break
    for point_x in range(w-1, 0, -1):
        if img_gray[point_y, point_x] > 10:
            RowPoints["whiteright"] = [point_x, point_y]
            break
    return RowPoints
def voc_crop(img_src, xml_src, img_save, xml_save):
    # Loop through each image in the dataset
    for image_file in os.listdir(img_src):
        # Load the image
        image_path = os.path.join(img_src, image_file)
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

        RowPoints = getRowWhitePoint(image_gray)
        x_cut_l = RowPoints["whiteleft"][0]
        x_cut_r = RowPoints["whiteright"][0]
        # Perform image cropping
        cropped_image = image[:, x_cut_l:x_cut_r]

        # Save the cropped image
        cropped_image_path = os.path.join(img_save, image_file)
        cv2.imwrite(cropped_image_path, cropped_image)

        # Load the corresponding XML file
        xml_file = os.path.join(xml_src, image_file[:-4] + ".xml")
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Update the XML file with the new image dimensions
        root.find("size/width").text = str(cropped_image.shape[1])
        root.find("size/height").text = str(cropped_image.shape[0])

        # 遍历 object
        for obj in root.iter('object'):  # 获取object节点中的name子节点
            xmin = obj.find('bndbox/xmin')
            xmax = obj.find('bndbox/xmax')
            x0 = int(xmin.text)-x_cut_l
            if x0 <= 0:
                x0 = 1
                print(image_file)
            obj.find('bndbox/xmin').text = str(x0)
            obj.find('bndbox/xmax').text = str(int(xmax.text)-x_cut_l)

        # Save the updated XML file
        updated_xml_file = os.path.join(xml_save, image_file[:-4] + ".xml")
        tree.write(updated_xml_file)

if __name__ == '__main__':
    root_dir = r"D:\04DataSets\02noname\01qipao_only/"
    img_src = root_dir + r"VOCdevkit\VOC2007/JPEGImages/"
    xml_src = root_dir + r"VOCdevkit\VOC2007/Annotations/"
    img_save = root_dir + r"VOCdevkit_small\VOC2007/JPEGImages/"
    xml_save = root_dir + r"VOCdevkit_small\VOC2007/Annotations/"
    voc_crop(img_src, xml_src, img_save, xml_save)





