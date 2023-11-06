import os, shutil
import re

def HeBingDir():
    # 遍历最外层
    for dir_class in os.listdir(src_dir):
        class_dir = os.path.join(src_dir, dir_class)
        if not os.path.isdir(class_dir):
            continue
        print(dir_class)
        class_name = get_class(dir_class)
        dst_class_dir = dst_dir + str(class_name) + "/"
        if os.path.exists(dst_class_dir):
            shutil.rmtree(dst_class_dir)
        os.mkdir(dst_class_dir)
        src = class_dir
        dst = dst_class_dir
        for item in os.listdir(src):
            file_path = os.path.join(src, item)
            if os.path.isdir(file_path):
                # 筛选
                if "确认" in file_path:
                    continue
                recursive_copy(file_path, dst)
            else:
                shutil.copy2(file_path, dst)


def recursive_copy(src, dst):
    for item in os.listdir(src):
        file_path = os.path.join(src, item)
        if os.path.isdir(file_path):
            recursive_copy(file_path, dst)
        else:
            shutil.copy2(file_path, dst)

def get_class(s):
    match = re.match(r"(\d+)", s)
    if match:
        number = int(match.group(1))
        return number

if __name__ == '__main__':
    dir_root = r"D:\02dataset\06淮河科技瑕疵分类\NG小图\/"
    src_dir = dir_root
    dst_dir = r"D:\02dataset\06HHKJ\HHKJ_not1030/"
    # dst_dir = r"D:\02dataset\06HHKJ\HHKJ1009\/"

    HeBingDir()