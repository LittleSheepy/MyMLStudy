import os, shutil,sys,getopt,random, tqdm

def move():
    for file in os.listdir(file_dir):
            # fileout = file.replace("微信图片_","")
            # if file != fileout:
        shutil.copyfile(file_dir + file, out_dir + file[:-3]+"png")
        #os.remove(file_dir + file)

def move1():
    for file in os.listdir(file_dir):
        in_file_dir = os.path.join(file_dir, file) + "/"
        out_file_dir = os.path.join(out_dir, file) + "/"
        if not os.path.exists(out_file_dir):
            os.mkdir(out_file_dir)
        for fi in os.listdir(in_file_dir):
            fi_en = fi[3:]
            shutil.copyfile(in_file_dir + fi, out_file_dir + fi_en)

if __name__ == '__main__':
    root_dir = r"D:/"
    file_dir = root_dir + r"051820_051908/"
    out_dir = root_dir + r"051820_051908_EN/"
    move1()