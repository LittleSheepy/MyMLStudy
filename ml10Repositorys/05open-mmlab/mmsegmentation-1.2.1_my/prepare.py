import argparse
import os
import re
import shutil
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from dsdl.converter.mllm import generate_config_file
from dsdl.converter.mllm import generate_readme_with_middle_format
from dsdl.converter.mllm import generate_readme_without_middle_format
from dsdl.converter.mllm import generate_tree_string
from dsdl.converter.utils import check_dsdl_meta_info
from dsdl.converter.utils import check_task_template_file
from dsdl.converter.utils import generate_class_dom
from dsdl.converter.utils import generate_subset_yaml_and_json
from dsdl.converter.utils import replace_special_characters

from tqdm import tqdm

def get_subset_samples_list(root_path, subset_name):
    root_path = Path(root_path)
    pattern = re.compile(r"^" + root_path.as_posix() + r"/")

    img_path_list = list(root_path.glob("*.jpg"))

    mask_path_list1 = [i.parent / (i.stem + "_1stHO.png") for i in img_path_list]
    mask_path_list2 = [i.parent / (i.stem + "_2ndHO.png") for i in img_path_list]

    samples_list = [{
        "image": pattern.sub('', image.as_posix()),
        "label_map_1stHO": pattern.sub('', mask1.as_posix()),
        "label_map_2ndHO": pattern.sub('', mask2.as_posix()),
    } for mask1, mask2, image in zip(mask_path_list1, mask_path_list2, img_path_list)]
    print(f"{subset_name}: {len(samples_list)}")
    return samples_list

def dataset_to_middle_format(root_path):
    root_path = Path(root_path)
    label_list = list(root_path.rglob("*.png"))
    multiprocess = True

    if not multiprocess:
        for label_path in tqdm(label_list):
            convertLabel(label_path)
    else:
        pool = Pool(max(cpu_count() - 4, 1))
        process = pool.imap_unordered(convertLabel, label_list)
        for p in tqdm(process, total=len(label_list)):
            ...

def convert(label):
    label = label.astype(np.uint8)
    return label

def convertLabel(label_path):
    label = np.asarray(Image.open(label_path)).copy()
    label = convert(label)
    Image.fromarray(label).save(label_path)

def parse_args():
    
    parse = argparse.ArgumentParser(
        description='Prepare the dsdl_SemSeg_full dataset from original dataset.'
    )
    parse.add_argument(
        '--decompressed', '-d', action='store_true',
        help='This argument decides whether the dataset files are decompressed. '
             'Add "-d" argument to skip decompress process, '
             'and directly pass the decompressed dataset. '
             'The default is need decompress process.'
    )
    parse.add_argument(
        '--copy', '-c', action='store_true',
        help='This argument decides whether the decompressed dataset files will be copied as a backup and then run the converter. '
             'Add "-c" argument to create a copy, and then run the converter. '
             'The default is not to create a copy and directly overwrite the original data.'
    )
    parse.add_argument(
        '--path', type=str,
        default=r"data/OpenDataLab___CHASE_DB1/\sample\image/",
        help='The original dataset path, a folder with compressed files if "-d" doesn\'t exist, '
             'or decompressed folder when "-d" exists.'
    )
    args = parse.parse_args()
    return args

def prepare(args):
    
    SCRIPT_PATH = Path(__file__).parent
    DSDL_PATH = SCRIPT_PATH.parent
    if args.decompressed:
        ORIGINAL_PATH = Path(args.path)
        if args.copy:
            PREPARED_PATH = ORIGINAL_PATH.parent / "prepared"
            if PREPARED_PATH.exists():
                raise Exception(f"Path {PREPARED_PATH.as_posix()} already exists.")
            shutil.copytree(ORIGINAL_PATH, PREPARED_PATH)
        else:
            if flag_middle_format:
                print("The operation will directly overwrite the dataset files with no backup.")
                while True:
                    confirm = input("Input [yes] to continue, or [quit] to exit.  ")
                    if confirm.lower() == "quit":
                        sys.exit(0)
                    elif confirm.lower() == "yes":
                        break
            PREPARED_PATH = ORIGINAL_PATH
    else:
        COMPRESSED_PATH = Path(args.path)
        PREPARED_PATH = COMPRESSED_PATH.parent / "prepared"
        if PREPARED_PATH.exists():
            raise Exception(f"Path {PREPARED_PATH.as_posix()} already exists.")

        for file in COMPRESSED_PATH.rglob("*.zip"):
            os.system(f'unzip -q "{file.as_posix()}" -d "{PREPARED_PATH.as_posix()}"')
        for file in COMPRESSED_PATH.rglob("*.tar.gz"):
            os.system(f'tar xf "{file.as_posix()}" -C "{PREPARED_PATH.as_posix()}"')

        if args.copy:
            ORIGINAL_PATH = COMPRESSED_PATH.parent / "original"
            if ORIGINAL_PATH.exists():
                raise Exception(f"Path {ORIGINAL_PATH.as_posix()} already exists.")
            shutil.copytree(PREPARED_PATH, ORIGINAL_PATH)

    return PREPARED_PATH.as_posix(), DSDL_PATH.as_posix()

if __name__ == "__main__":

    meta_info = {
        "Dataset Name": "CHASE_DB1",
        "HomePage": "https://blogs.kingston.ac.uk/retinal/chasedb1/",
        "Modality": "Images",
        "Task": "Semantic Segmentation"
    }

    flag_middle_format = True  
    class_dom_names_original = ['retinal']
    subset_name_list = ['train']

    args = parse_args()
    root_path, save_path = prepare(args)

    check_dsdl_meta_info(meta_info)  
    check_task_template_file(save_path)  

    original_tree_str = generate_tree_string(root_path)

    if flag_middle_format:

        dataset_to_middle_format(root_path)  

    class_dom_names = []
    for name in class_dom_names_original:
        class_dom_names.append(replace_special_characters(name))  
    generate_class_dom(save_path, class_dom_names)  

    for subset_name in subset_name_list:
        meta_info["Subset Name"] = subset_name
        print(f"processing data in {subset_name}.")
        subset_samples_list = get_subset_samples_list(root_path, subset_name)  

        generate_subset_yaml_and_json(
            meta_info,
            save_path,
            subset_samples_list
        )
        print(f"Sample list for {subset_name} is generated.")
    dsdl_tree_str = generate_tree_string(save_path,display_num=100)  

    generate_config_file(save_path)  

    if flag_middle_format:
        generate_readme_with_middle_format(
            save_path,
            meta_info["Dataset Name"],
            meta_info["Task"],
            original_tree_str,
            dsdl_tree_str,
        )
    else:
        generate_readme_without_middle_format(
            save_path,
            meta_info["Dataset Name"],
            meta_info["Task"],
            original_tree_str,
            dsdl_tree_str
        )
