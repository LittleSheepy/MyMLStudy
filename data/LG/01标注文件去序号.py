import os
import shutil


def move_num(src_dir, tar_dir):
    for filename in os.listdir(src_dir):
        if '-' in filename:
            base, ext = os.path.splitext(filename)
            new_base = base.split('-', 1)[0]
            new_filename = new_base + ext
        else:
            new_filename = filename
        shutil.copy(os.path.join(src_dir, filename), os.path.join(tar_dir, new_filename))
        # os.rename(os.path.join(src_dir, filename), os.path.join(tar_dir, new_filename))

def main():
    if os.path.exists(src_source):
        os.makedirs(tar_source)
        move_num(src_source, tar_source)
    if os.path.exists(src_label):
        os.makedirs(tar_label)
        move_num(src_label, tar_label)


if __name__ == '__main__':
    dir_root = r"E:\01项目\01LG项目\02数据\01阿丘标注数据\07内面油污\/"
    src_name = r"标注_非训练集"
    src_root = dir_root + src_name + "/"
    src_source = src_root + r"source"
    src_label = src_root + r"label"

    tar_root = dir_root + src_name + "_result" + "/"
    tar_source = tar_root + r"source"
    tar_label = tar_root + r"label"
    if not os.path.exists(tar_root):
        os.makedirs(tar_root)
    main()




















