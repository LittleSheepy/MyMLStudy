import os, shutil



def move1():
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect/predict11/"
    root_dir = r"D:\03GitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict14\/"
    root_dir = r"F:\0LG标注_yolo\12-侧面黑色顶底加强筋破损\/"
    img_dir = root_dir + r"/imgOK/"
    biaozhu = root_dir + "/img/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/txt/"
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
    root_dir = r"D:\04Bin2\/"
    img_dir = root_dir + r"/images/"
    biaozhu = root_dir + "/None/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = root_dir + "/image_result/"
    for xmlfile in os.listdir(xml_dir):
        imgfile = xmlfile[:-4] + ".jpg"
        if os.path.exists(img_dir + imgfile):
            shutil.move(img_dir + imgfile, biaozhu + imgfile)
            # shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
        else:
            print(imgfile)



if __name__ == '__main__':
    move1()