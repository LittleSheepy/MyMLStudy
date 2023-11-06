import os, shutil


def txtCopy():
    with open(txt_file, 'r',encoding="utf-8") as file:
        for line in file:
            line = line[:-1]
            if not os.path.exists(src_file + "/" + line):
                continue
            # shutil.copyfile(src_file + "/" + line, dst_file + "/" + line)
            shutil.move(src_file + "/" + line, dst_file + "/" + line)



if __name__ == '__main__':
    dir_root = r"D:\02dataset\06淮河科技瑕疵分类\NG小图\/"
    # dir_root = r"D:\02dataset\06HHKJ\HHKJ1009\/"
    src_file = dir_root + r"\\9涂层脱落\9涂层脱落零一/"
    dst_file = dir_root + r"/got"
    txt_file = dir_root + r"/got.txt"
    if not os.path.exists(dst_file):
        os.mkdir(dst_file)
    txtCopy()
