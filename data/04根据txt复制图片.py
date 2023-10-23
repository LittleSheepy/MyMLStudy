import os, shutil


def txtCopy():
    with open(txt_file, 'r',encoding="utf-8") as file:
        for line in file:
            line = line[:-1]
            shutil.copyfile(src_file + "/" + line, dst_file + "/" + line)



if __name__ == '__main__':
    dir_root = r"D:\02dataset\01work\06淮河科技瑕疵分类\瑕疵小图-AI训练用\NG\NG小图\/"
    src_file = dir_root + r"/脏污"
    dst_file = dir_root + r"/脏污1"
    txt_file = dir_root + r"/脏污.txt"
    txtCopy()
