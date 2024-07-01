import os, shutil

def main():
    for dirname in os.listdir(NG_dir):
        if "(2)" in dirname:
            pos = dirname.find("(2)")
            new_name = dirname[:pos]
            sub_dir = root_dir + "/2/"
            shutil.copyfile(os.path.join(NG_dir, dirname), os.path.join(sub_dir, new_name + ".jpg"))
        elif "(3)" in dirname:
            pos = dirname.find("(3)")
            new_name = dirname[:pos]
            sub_dir = root_dir + "/3/"
            shutil.copyfile(os.path.join(NG_dir, dirname), os.path.join(sub_dir, new_name + ".jpg"))
        elif "(4)" in dirname:
            pos = dirname.find("(4)")
            new_name = dirname[:pos]
            sub_dir = root_dir + "/4/"
            shutil.copyfile(os.path.join(NG_dir, dirname), os.path.join(sub_dir, new_name + ".jpg"))
        else:
            pos = dirname.find("_CM")
            new_name = dirname[:pos+6]
            sub_dir = root_dir + "/1/"
            shutil.copyfile(os.path.join(NG_dir, dirname), os.path.join(sub_dir, new_name + ".jpg"))



if __name__ == '__main__':
    root_dir = r"G:\黑灰合并侧面破损结果图 0523\8线左/"
    NG_dir = root_dir + r"NG/"
    for i in range(4):
        sub_dir = root_dir + "/" + str(i+1)
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
    main()
