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


BianLiDirToTxt()