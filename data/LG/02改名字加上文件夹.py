import os, shutil


def rename(sub_dir, prefix=""):
    for dirname in os.listdir(sub_dir):
        if os.path.isdir(os.path.join(sub_dir, dirname)):
            if not "_" in dirname:
                prefix_new = prefix + "_" + dirname
            if "black_0534006" in dirname:
                continue
            rename(os.path.join(sub_dir, dirname), prefix_new)
        else:
            if "CM" in dirname:
                shutil.copyfile(os.path.join(sub_dir, dirname), os.path.join(save_dir, prefix + "_" + dirname))

def main():
    for dirname in os.listdir(src_dir):
        rename(os.path.join(src_dir, dirname), dirname)

if __name__ == '__main__':
    root_dir = r"E:\0ProjectData\0LG_CB_DATA\10测试数据\点检四方向/"
    src_dir = root_dir + r"原图/"
    save_dir = root_dir + r"原图_NM改名_jpg/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    main()
