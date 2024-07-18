import os
from PIL import Image


def crop_center(image, crop_size):
    w, h = image.size
    left = (w - crop_size) // 2
    top = (h - crop_size) // 2
    right = (w + crop_size) // 2
    bottom = (h + crop_size) // 2
    return image.crop((left, top, right, bottom))


def process_images(input_folder, output_folder, crop_size=500):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        img = Image.open(file_path)

        if img is not None:
            w, h = img.size
            if h >= crop_size and w >= crop_size:
                cropped_img = crop_center(img, crop_size)
                output_path = os.path.join(output_folder, filename)
                cropped_img.save(output_path)

if __name__ == '__main__':
    # dir_root = r"D:\02dataset\01work\09DGKaiDe\00imgAll/"
    dir_root = r"./"
    # dir_root = r"D:\02dataset\01work\07HZHengTai\00imgAll1\01ZhengJi/"
    dir_root = r"D:\02dataset\01work\09DGKaiDe\00imgAll\06TaiGun_result\/"

    input_folder = dir_root + '03/'
    output_folder = dir_root + '00_500/'
    process_images(input_folder, output_folder)