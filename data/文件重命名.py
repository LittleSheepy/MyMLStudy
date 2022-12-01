import os, shutil,sys,getopt,random

file_dir = r"D:\01sheepy\01work\07xiling\out1/jpg/"
out_dir = r"D:\01sheepy\01work\07xiling\out1/png/"
for file in os.listdir(file_dir):
        # fileout = file.replace("微信图片_","")
        # if file != fileout:
    shutil.copyfile(file_dir + file, out_dir + file[:-3]+"png")
    #os.remove(file_dir + file)