import os
import cv2

source_folder = r'E:\0ProjectData\0LG_CB_DATA\2labelData\01NM_LZPS\1all\\mask/'
destination_folder = r'E:\0ProjectData\0LG_CB_DATA\2labelData\01NM_LZPS\1all\\jpg/'

for filename in os.listdir(source_folder):
    img = cv2.imread(os.path.join(source_folder, filename))
    if img is not None:
        img = img * 128
        cv2.imwrite(os.path.join(destination_folder, filename), img)
