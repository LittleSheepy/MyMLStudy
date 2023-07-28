import os, shutil

def aq_num2name(xml_file):
    f = open(os.path.join(xml_file), "r")


def aq2name():
    for dir_ in os.listdir(dir_biaozhu):
    #for dir_ in dirlist:
        dir_num = dir_.split("-")[0]
        fullpath_ = os.path.join(dir_biaozhu, dir_)
        fullpath_name = os.path.join(dir_name, dir_)
        if not os.path.isdir(fullpath_):
            continue
        if not os.path.exists(fullpath_name):
            os.mkdir(fullpath_name)
        db_dir = root_db + dir_
        bd_path = None
        for dir_db in os.listdir(db_dir):
            if dir_db[-2:] == "db":
                bd_path = os.path.join(db_dir, dir_db)
        f = open(os.path.join(bd_path), "r", encoding='utf-8', errors='ignore')
        lines = f.readlines()
        line_cnt = len(lines)
        line_i = 0
        while True:
            if line_i >= line_cnt:
                break
            if "source_image_path" in lines[line_i]:
                str_source = lines[line_i][:-1]
                path = str_source.split(":")[1].strip()  # 去除前后空格
                path = path.split(",")[0].strip()  # 去除前后空格
                filename_source = path.split("/")[-1]  # 获取最后一个"/"后的部分
                filename_source = filename_source[:-1]

                str_storage = lines[line_i+1][:-1]

                if not filename_source[-3:] in str_storage:
                    line_i = line_i + 1
                    continue
                path = str_storage.split(":")[1].strip()  # 去除前后空格
                filename_storage = path.split("/")[-1]  # 获取最后一个"/"后的部分
                # etx = filename_storage[-6:-2]
                # filename_storage_name = filename_storage.split("-")[0]
                # filename_storage_save = filename_storage_name + etx
                # filename_storage_save = filename_storage_save[1:]
                filename_storage_save = filename_storage[1:-2]
                print(filename_source, filename_storage_save)

                filename_storage_save = "aidi" + dir_num + "_"
                filename_source_full = os.path.join(fullpath_, filename_source)
                filename_storage_save_full = os.path.join(fullpath_name, filename_storage_save)
                filename_source_full_json = filename_source_full[:-4] + ".json"
                filename_storage_save_full_json = filename_storage_save_full[:-4] + ".json"
                if os.path.exists(filename_source_full):
                    shutil.move(filename_source_full, filename_storage_save_full)
                    if os.path.exists(filename_source_full_json):
                        shutil.move(filename_source_full_json, filename_storage_save_full_json)
                else:
                    print(filename_source)

                line_i = line_i + 2

            else:
                line_i = line_i + 1


def aq2name_bynum():
    for dir_ in os.listdir(dir_biaozhu):
        dir_num = dir_.split("-")[0]
        fullpath_ = os.path.join(dir_biaozhu, dir_)
        fullpath_name = os.path.join(dir_name, dir_)
        if not os.path.isdir(fullpath_):
            continue
        if not os.path.exists(fullpath_name):
            os.mkdir(fullpath_name)
        for filename in os.listdir(fullpath_):
            # if filename[-4:] == "json":
            #     continue
            filename_storage_save = "aidi" + dir_num + "_" + filename
            filename_source_full = os.path.join(fullpath_, filename)
            filename_storage_save_full = os.path.join(fullpath_name, filename_storage_save)
            shutil.move(filename_source_full, filename_storage_save_full)

            # filename_source_full_json = filename_source_full[:-4] + ".json"
            # filename_storage_save_full_json = filename_storage_save_full[:-4] + ".json"
            # if os.path.exists(filename_source_full_json):
            #     shutil.move(filename_source_full_json, filename_storage_save_full_json)


def aq2name_bynum2():
    for filename in os.listdir(dir_biaozhu):
        # if filename[-4:] == "json":
        #     continue
        filename_storage_save = "aidi16" + "_" + filename
        filename_source_full = os.path.join(dir_biaozhu, filename)
        filename_storage_save_full = os.path.join(dir_name, filename_storage_save)
        shutil.move(filename_source_full, filename_storage_save_full)

if __name__ == '__main__':
    root_dir = r"E:\0ProjectData\0LG_CB_DATA\1AIDI_TrainData\0LG_label_name\BM\/"
    root_db = r"H:\15project\02kd\03LG\AIDI\所有模型\/"
    dir_biaozhu = root_dir + "16BM_JQJPS/"
    dir_name = root_dir + "16BM_JQJPS_name/"
    dirlist = ["8-反面黑色加强筋破损"]
    aq2name_bynum2()

