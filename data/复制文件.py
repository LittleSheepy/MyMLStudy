import os, shutil

def main():
    img_dir = r"D:\01sheepy\01work\06ningbo\01img\imgTrain\img_wu/"
    xml_dir = r"D:\01sheepy\01work\06ningbo\01img\imgTrain\xml_wu/"
    for imgfile in os.listdir(img_dir):
        shutil.copyfile(xml_dir + "demo.xml", xml_dir + imgfile[:-4]+".xml")

if __name__ == '__main__':
    main()