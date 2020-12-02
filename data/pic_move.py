import os, shutil







def move1():
    img_dir = "D:/01sheepy/01work/01baojie_ocr/pp/img/"
    biaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_biaozhu/"
    nobiaozhu = "D:/01sheepy/01work/01baojie_ocr/pp/img_nobiaozhu/"
    xml_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml/"
    for imgfile in os.listdir(img_dir):
        moveFlg = False
        for xmlfile in os.listdir(xml_dir):
            print(xmlfile)
            if xmlfile[:-4] == imgfile[:-4]:
                shutil.copyfile(img_dir + imgfile, biaozhu + imgfile)
                moveFlg = True
        if not moveFlg:
            shutil.copyfile(img_dir + imgfile, nobiaozhu + imgfile)



if __name__ == '__main__':
    move1()