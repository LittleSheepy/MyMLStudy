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
    root_dir = r"D:\05xxDMLBPS\07xxx\/"
    img_dir = root_dir + r"img_ok/"
    biaozhu = root_dir + "img/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/txt/"
    if not os.path.exists(biaozhu):
        os.mkdir(biaozhu)
    for xmlfile in os.listdir(xml_dir):
        imgfile = xmlfile[:-4] + ".jpg"
        if os.path.exists(img_dir + imgfile):
            shutil.move(img_dir + imgfile, biaozhu + imgfile)
            #shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
            print("have ", imgfile)
        else:
            pass
            # print(imgfile)

if __name__ == '__main__':
    moveByLabel()