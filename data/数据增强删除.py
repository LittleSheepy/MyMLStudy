import os




def main():
    fileorg_dor = r"D:\01sheepy\01work\06ningbo\01img\00上一个项目缺陷\img/"
    fileaug_dor = r"D:\01sheepy\01work\06ningbo\ninggangnew_quan\JPEGImages/"
    i = 0
    for file_org in os.listdir(fileorg_dor):
        i = i + 1
        print(i, ",", file_org)
        for file_aug in os.listdir(fileaug_dor):
            if file_aug[:-4] == file_org[:-4]:
                os.remove(fileaug_dor+"/"+file_aug)
#1N1109531020_S1_C02_P00004000916
#1N1109531020_S1_C02_P00017.jpg
def delname_len():
    fileaug_dor = r"D:\01sheepy\01work\06ningbo\01img\01第一批宁波数据\xml\/"
    for file_aug in os.listdir(fileaug_dor):
        if len(file_aug) > 34:
            os.remove(fileaug_dor + "/" + file_aug)
delname_len()