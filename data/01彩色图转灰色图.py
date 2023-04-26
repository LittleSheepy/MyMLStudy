from PIL import Image
import os
#
# # 指定原始图片文件夹路径和目标文件夹路径
# original_folder = '/path/to/original/folder'
# target_folder = '/path/to/target/folder'
#
# # 遍历原始文件夹中的所有图片文件
# for filename in os.listdir(original_folder):
#     if filename.endswith('.jpg') or filename.endswith('.png'):
#         # 打开图片文件
#         img = Image.open(os.path.join(original_folder, filename))
#         # 转换为灰度图
#         gray_img = img.convert('L')
#         # 生成目标文件路径
#         target_path = os.path.join(target_folder, filename)
#         # 保存灰度图
#         gray_img.save(target_path)

def to_gray():
    for dir_name in os.listdir(original_folder):
        original_folder_child = original_folder + dir_name
        target_folder_child = target_folder + dir_name
        os.mkdir(target_folder_child)
        # 遍历原始文件夹中的所有图片文件
        for filename in os.listdir(original_folder_child):
            if filename.endswith('.bmp') or filename.endswith('.png'):
                # 打开图片文件
                img = Image.open(os.path.join(original_folder_child, filename))
                # 转换为灰度图
                gray_img = img.convert('L')
                # 生成目标文件路径
                target_path = os.path.join(target_folder_child, filename)
                # 保存灰度图
                gray_img.save(target_path)


if __name__ == '__main__':
    original_folder = r'C:\Users\11658\Desktop\水渍图像/'
    target_folder = r'C:\Users\11658\Desktop\水渍图像_灰色/'

    to_gray()
    print("完成")