import os
import cv2

# 输入和输出文件夹路径
dir_root = r"D:\02dataset\01work\11OCR\03lableimg_one/"
input_images_folder = dir_root + 'img_OK/'
txt_words_folder = dir_root + 'txt_words'
txt_char_folder = dir_root + 'txt_char'
txt_little_char_folder = dir_root + 'txt_little_char'
output_folder = dir_root + 'img_little'

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 遍历所有标注文件
for label_file in os.listdir(txt_words_folder):
    if label_file.endswith('.txt'):
        if "classes" in label_file:
            continue
        # 读取标注文件
        with open(os.path.join(txt_words_folder, label_file), 'r') as f:
            lines = f.readlines()

        # 获取对应的图像文件
        image_file = label_file.replace('.txt', '.jpg')  # 假设图像为 .jpg 格式
        image_path = os.path.join(input_images_folder, image_file)
        image = cv2.imread(image_path)

        # 遍历每一行标注
        line = lines[0]
        class_id, x_center, y_center, width, height = map(float, line.split())
        # 计算边界框的坐标
        x1 = int((x_center - width / 2) * image.shape[1])
        y1 = int((y_center - height / 2) * image.shape[0])
        x2 = int((x_center + width / 2) * image.shape[1])
        y2 = int((y_center + height / 2) * image.shape[0])

        # 剪裁图像
        cropped_image = image[y1:y2, x1:x2]

        # 保存剪裁后的图像
        output_file = os.path.join(output_folder, image_file)
        cv2.imwrite(output_file, cropped_image)

        # 保存小框

        # 读取标注文件
        with open(os.path.join(txt_char_folder, label_file), 'r') as f:
            lines = f.readlines()

        f_txt_little_char = open(os.path.join(txt_little_char_folder, label_file), 'a')
        w = x2 - x1
        h = y2 - y1
        for line in lines:
            class_id, x_center, y_center, width, height = map(float, line.split())
            # 计算边界框的坐标
            lx_c = int(x_center * image.shape[1]) - x1
            ly_c = int(y_center * image.shape[0]) - y1
            lw = int(width * image.shape[1])
            lh = int(height * image.shape[0])

            plx_c = round(lx_c / w, 6)
            ply_c = round(ly_c / h, 6)
            plw = round(lw / w, 6)
            plh = round(lh / h, 6)
            line_new = str(int(class_id)) + " "
            line_new = line_new + str(plx_c) + " "
            line_new = line_new + str(ply_c) + " "
            line_new = line_new + str(plw) + " "
            line_new = line_new + str(plh) + "\n"
            f_txt_little_char.write(line_new)
        f_txt_little_char.close()