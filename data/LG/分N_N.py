import os, shutil



def moveNN():
    root_dir = r"./"

    for file in os.listdir(root_dir):
        file_full_path = os.path.join(root_dir, file)
        if not os.path.isfile(file_full_path):
            continue

        file_split = file.split(".")
        fileName = file_split[0]
        ext = file_split[1]
        if not ext in ["jpg", "bmp", "png"]:
            continue
        nn = fileName[-3:]
        nn_path = os.path.join(root_dir, nn)
        if not os.path.exists(nn_path):
            os.mkdir(nn_path)

        shutil.move(file_full_path, os.path.join(root_dir, nn, file))



if __name__ == '__main__':
    moveNN()