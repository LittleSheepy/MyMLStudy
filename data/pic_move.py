import os, shutil



def move1():
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect/predict11/"
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict14\/"
    root_dir = r"D:\05xx\0518\/"
    img_dir = root_dir + r"NG\/"
    biaozhu = root_dir + "/NG1/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/img/"
    for imgfile in os.listdir(img_dir):
        moveFlg = False
        for xmlfile in os.listdir(xml_dir):
            print(xmlfile)
            if xmlfile[:-4] == imgfile[:-4]:
                # shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                shutil.move(img_dir + imgfile, biaozhu + imgfile)
                moveFlg = True
        # if not moveFlg:
        #     shutil.copyfile(img_dir + imgfile, nobiaozhu + imgfile)

def moveByLabel():
    root_dir = r"D:\05xx\/"
    img_dir = root_dir + r"/test/"
    biaozhu = root_dir + "/新建文件夹/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/img_val/"
    for xmlfile in os.listdir(xml_dir):
        imgfile = xmlfile[:-4] + ".jpg"
        if os.path.exists(img_dir + imgfile):
            shutil.move(img_dir + imgfile, biaozhu + imgfile)
            #shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
        else:
            print(imgfile)



if __name__ == '__main__':
    move1()