import cv2
import os

source_folder = r"D:\02dataset\01work\05nanjingLG\07aidi\01AIDI_SZ\aidi\01labeled_all\source/"
target_folder = r"D:\02dataset\01work\05nanjingLG\07aidi\01AIDI_SZ\img_no/"

for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith(".png"):
            # Load the BMP file
            bmp_path = os.path.join(root, file)
            img = cv2.imread(bmp_path)

            # Convert to JPG and save to target folder
            jpg_path = os.path.join(target_folder, os.path.basename(root), file.replace(".png", ".jpg"))
            cv2.imwrite(jpg_path, img)
