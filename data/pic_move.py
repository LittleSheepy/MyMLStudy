import os, shutil



def move1():
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect/predict11/"
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict14\/"
    root_dir = r"D:\0\0LG_DATA\SZ_OKNG_0607\/"
    img_dir = root_dir + r"/img_save2_1/"
    biaozhu = root_dir + "/img_save_have0608_8/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/labels/"
    for imgfile in os.listdir(img_dir):
        moveFlg = False
        for xmlfile in os.listdir(xml_dir):
            print(xmlfile)
            if xmlfile[:-4] == imgfile[:-4]:
                # shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                moveFlg = True
        # if not moveFlg:
        #     shutil.copyfile(img_dir + imgfile, nobiaozhu + imgfile)



if __name__ == '__main__':
    move1()