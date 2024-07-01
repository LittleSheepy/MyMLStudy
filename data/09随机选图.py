import os
import random
import shutil


def copy_random_images(src_folder, dst_folder, num_images=100):
    # 获取源文件夹中的所有文件
    all_files = os.listdir(src_folder)
    # 过滤出图像文件（假设图像文件以常见的图像扩展名结尾）
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    # 随机选取指定数量的图像文件
    selected_images = random.sample(image_files, min(num_images, len(image_files)))

    # 确保目标文件夹存在
    os.makedirs(dst_folder, exist_ok=True)

    # 复制选中的图像文件到目标文件夹
    for image in selected_images:
        src_path = os.path.join(src_folder, image)
        dst_path = os.path.join(dst_folder, image)
        shutil.copy(src_path, dst_path)

def one_dir_copy_random_images():
    src_folder = dir_root + r'08ZheZhou_result\08/'
    dst_folder = dir_root + r'08/'
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)
    copy_random_images(src_folder, dst_folder)

def dirs_copy_random_images():
    for dir_name in os.listdir(dir_root):
        print(dir_name)
        if not "result" in dir_name:
            continue
        BianHao = dir_name[:2]
        src_folder = os.path.join(dir_root, dir_name, BianHao)
        dst_folder = os.path.join(dir_root, BianHao)
        copy_random_images(src_folder, dst_folder)

if __name__ == '__main__':
    dir_root = r"D:\02dataset\01work\07HZHengTai\00imgAll1\02FuJi/"
    # dir_root = r"D:\02dataset\01work\07HZHengTai\00imgAll1\01ZhengJi/"
    # dir_root = r"D:\02dataset\01work\07HZHengTai\00imgAll/"

    dirs_copy_random_images()
