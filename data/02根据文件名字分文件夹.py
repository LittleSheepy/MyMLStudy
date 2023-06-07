import os

def move():
    root_dir = r"D:\0\0LG_DATA\SZ_NG_0529\black\img_have0606\/"
    img_dir = root_dir + "/img_save11/"
    for imgfile in os.listdir(img_dir):
        moveFlg = False


if __name__ == '__main__':
    move()