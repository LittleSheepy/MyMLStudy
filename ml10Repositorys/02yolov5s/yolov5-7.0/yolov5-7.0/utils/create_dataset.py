




































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
        train, test = train_test_split(listdir, test_size=test_per/100, shuffle=True, random_state=0)
        train, val = train_test_split(train, test_size=val_per/100, shuffle=True, random_state=0)

        os.makedirs(os.path.join(data_path_new, "train", str(i)))
        os.makedirs(os.path.join(data_path_new, "val", str(i)))
        os.makedirs(os.path.join(data_path_new, "test", str(i)))
        for name in train:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/train/{}/{}'.format(data_path_new, i, name))

        for name in val:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/val/{}/{}'.format(data_path_new, i, name))

        for name in test:
            shutil.copy('{}/{}'.format(dir_src_class, name), '{}/test/{}/{}'.format(data_path_new, i, name))






