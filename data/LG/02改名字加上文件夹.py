import os


def rename(sub_dir, prefix=""):
    for dirname in os.listdir(sub_dir):
        if os.path.isdir(os.path.join(sub_dir, dirname)):
            if not "_" in dirname:
                prefix = prefix + dirname + "_"
            rename(os.path.join(sub_dir, dirname), prefix)
        else:
            if "NM" in dirname:
                os.rename(os.path.join(sub_dir, dirname), os.path.join(save_dir, prefix + "_" + dirname))

def main():
    for dirname in os.listdir(root_dir):
        rename(os.path.join(root_dir, dirname), dirname+"_")

if __name__ == '__main__':
    root_dir = r"E:\点检和测试\点检图片\整个托盘\缺陷补充后的点检托盘原图/"
    save_dir = r"E:\点检和测试\点检图片\整个托盘\NM/"
    main()
