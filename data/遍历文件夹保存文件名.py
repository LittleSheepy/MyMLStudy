import os
"""
    --dir_root
        --dir
        --result.txt
"""
def BianLiDirToTxt():
    dir_root = r"D:\01sheepy\01work\06ningbo\02images_quexian\\/"
    file_dir = dir_root+"/txt/"
    f = open(os.path.join(dir_root, "result.txt"), "a")
    for filename in os.listdir(file_dir):
        print(filename)
        f.write(filename+"\n")
    f.close()

# 统计文件名
def BianLiDirFilesToTxt():
    dir_root = r"D:\00myGitHub\aFolder_YOLO\03yolov8s\ultralytics\runs\detect\predict26_1_1_0.5/"
    file_dir = dir_root+"/labels/"
    f = open(os.path.join(dir_root, "result.txt"), "a")
    filenameList = []
    for filename in os.listdir(file_dir):
        name = filename[:-8]
        if name in filenameList:
            continue
        print(name)
        f.write(name+"\n")
        filenameList.append(name)
    print(len(filenameList))
    f.close()


BianLiDirFilesToTxt()