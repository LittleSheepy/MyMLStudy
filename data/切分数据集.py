import os
import random
import shutil

# 定义文件夹路径和切分比例

folder_path = r"D:\04DataSets\03myDataSet28/"
target_name = "03myDataSet28/"
img_end = r"\images\train2017/"
txt_end = r"\labels\train2017/"

img_src_dir = folder_path + img_end
txt_src_dir = folder_path + txt_end
img_dsc_dir = folder_path + img_end.replace("train2017", "val2017")
txt_dsc_dir = folder_path + txt_end.replace("train2017", "val2017")
os.makedirs(img_dsc_dir, exist_ok=True)
os.makedirs(txt_dsc_dir, exist_ok=True)
split_ratio = 0.7
# 读取文件夹下所有文件
file_list = os.listdir(img_src_dir)

# 打乱文件列表的顺序
random.shuffle(file_list)

# 计算切分点
split_index = int(len(file_list) * split_ratio)

# 切分文件列表
train_file_list = file_list[:split_index]
val_file_list = file_list[split_index:]

# 复制测试集文件到特定目录
for file in val_file_list:
    img_src_path = img_src_dir + file
    img_dst_path = img_dsc_dir + file
    shutil.move(img_src_path, img_dst_path)
    txt_src_path = txt_src_dir + file.replace("bmp", "txt")
    txt_dst_path = txt_dsc_dir + file.replace("bmp", "txt")
    shutil.move(txt_src_path, txt_dst_path)
