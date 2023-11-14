import os

def save_filenames_to_txt(directory, txt_file):
    with open(txt_file, 'w') as f:
        for filename in os.listdir(directory):
            base_name = os.path.splitext(filename)[0]
            f.write(base_name + '\n')


if __name__ == '__main__':
    dir_root = r"E:\0ProjectData\AI_PK_pictreue\1_Front_LZPS\voc\/"
    img_dir = dir_root + r"JPEGImages"
    txt_path = dir_root + "name.txt"
    save_filenames_to_txt(img_dir, txt_path)