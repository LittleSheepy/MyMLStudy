import os, shutil



def move1():
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect/predict11/"
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict14\/"
    root_dir = r"D:\05xxnmlbps\0811test/"
    img_dir = root_dir + r"NG/"
    biaozhu = root_dir + "have/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/img_aq/"
    if not os.path.exists(biaozhu):
        os.mkdir(biaozhu)
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
    root_dir = r"D:\02dataset\06淮河科技瑕疵分类\NG小图/"
    img_dir = root_dir + r"/got - 副本/"
    biaozhu = root_dir + "/重复/"
    xml_dir = root_dir + "/got/"
    # if not os.path.exists(biaozhu):
    #     os.mkdir(biaozhu)
    for xmlfile in os.listdir(xml_dir):
        imgfile = xmlfile[:-5] + ".jpeg"
        if os.path.exists(img_dir + imgfile):
            # shutil.move(img_dir + imgfile, biaozhu + imgfile)
            os.remove(img_dir + imgfile)
            #shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
            print("have ", imgfile)
        else:
            pass
            # print(imgfile)
        imgfile = xmlfile[:-4] + ".bmp"
        if os.path.exists(img_dir + imgfile):
            # shutil.move(img_dir + imgfile, biaozhu + imgfile)
            os.remove(img_dir + imgfile)
            #shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
            print("have ", imgfile)
        else:
            pass
            # print(imgfile)

if __name__ == '__main__':
    moveByLabel()