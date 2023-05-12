import cv2
import os

source_folder = r"C:\Users\11658\Desktop\a\b\src/"
target_folder = r"C:\Users\11658\Desktop\a\b\save/"

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".bmp"):
            # Load the BMP file
            bmp_path = os.path.join(root, file)
            img = cv2.imread(bmp_path)

            # Convert to JPG and save to target folder
            jpg_path = os.path.join(target_folder, os.path.basename(root), file.replace(".bmp", ".jpg"))
            cv2.imwrite(jpg_path, img)