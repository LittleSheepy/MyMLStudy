import sys, os
import shutil
from sklearn.model_selection import train_test_split


def create_dataset(src_dir, tar_dir, class_nums, val_per, test_per):
    date_name = os.path.basename(os.path.normpath(src_dir))
    data_path_new = os.path.join(tar_dir, date_name)
    if os.path.exists(data_path_new):
        shutil.rmtree(data_path_new)
    os.makedirs(data_path_new)
    os.makedirs(os.path.join(data_path_new, "train"))
    os.makedirs(os.path.join(data_path_new, "val"))
    os.makedirs(os.path.join(data_path_new, "test"))
    for i in range(class_nums):
        dir_src_class = os.path.join(src_dir, str(i))
        #dir_tar_class = os.path.join(data_path_new, str(i))
        listdir = [i for i in os.listdir(dir_src_class)]
        # 限制最大值
        if len(listdir) > 3000:
            listdir, test = train_test_split(listdir, test_size=1-3000/len(listdir), shuffle=True, random_state=0)


        if listdir == []:
            continue
        if test_per > 0:
            train, test = train_test_split(listdir, test_size=test_per/100, shuffle=True, random_state=0)
        else:
            train, test = listdir, []
        val = []
        if val_per > 0 and len(train) > 1:
            train, val = train_test_split(train, test_size=val_per/100, shuffle=True, random_state=0)

        os.makedirs(os.path.join(data_path_new, "train", str(i)))
        os.makedirs(os.path.join(data_path_new, "val", str(i)))
        os.makedirs(os.path.join(data_path_new, "test", str(i)))
        for name in train:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/train/{}/{}'.format(data_path_new, i, name))

        for name in val:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/val/{}/{}'.format(data_path_new, i, name))

        for name in val:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/test/{}/{}'.format(data_path_new, i, name))


if __name__ == '__main__':
    data_name = "HHKJ1103_3000"
    # data_name = "HHKJ_9_1030"
    # data_name = "HHKJ_all1030"
    root_dir = r"D:\02dataset\06HHKJ/"
    src_dir = root_dir + data_name + r"/"
    tar_dir = root_dir + data_name + r"_train/"
    class_nums = 12
    val_per = 30
    test_per = 0
    create_dataset(src_dir, tar_dir, class_nums, val_per, test_per)


