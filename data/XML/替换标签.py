import os
import xml.etree.ElementTree as ET

def main():
    xml_dir = r"D:\01sheepy\01work\06ningbo\01img\imgdata\imgTrain\xml/"
    xml_dir_out = r"D:\01sheepy\01work\06ningbo\01img\imgdata\imgTrain\xml1/"
    for xml_file in os.listdir(xml_dir):
        print(xml_file)
        # filename = os.path.join(xml_dir, xml_file)
        # tree = ET.parse(filename)
        # objs = tree.findall('object')
        f = open(os.path.join(xml_dir, xml_file), "r")
        f_out = open(os.path.join(xml_dir_out, xml_file), "a")
        data = f.read()
        data1 = data.replace("<name>gt</name>","<name>no</name>")
        f_out.write(data1)
        f.close()
        f_out.close()


if __name__ == '__main__':
    main()