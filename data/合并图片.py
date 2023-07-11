import os
from PIL import Image

# Define the directories
root_dir = r"D:\04Bin/"
dir1 = root_dir + 'NG'
dir2 = root_dir + 'result'
dir_save = root_dir + 'NG_RESULT'

# Get the list of image files
images1 = [i for i in os.listdir(dir1) if i.endswith('.jpg') or i.endswith('.png')]
images2 = [i for i in os.listdir(dir2) if i.endswith('.jpg') or i.endswith('.png')]

# Sort the images by name
images1.sort()
images2.sort()

# Ensure the two directories have the same number of images
assert len(images1) == len(images2)

# Iterate over the images
for i in range(len(images1)):
    # Open the images
    img1 = Image.open(os.path.join(dir1, images1[i]))
    img2 = Image.open(os.path.join(dir2, images2[i]))

    # Get the size of the images
    width1, height1 = img1.size
    width2, height2 = img2.size

    # Create a new image with the combined height of the two images
    new_img = Image.new('RGB', (max(width1, width2), height1 + height2))

    # Paste the images into the new image
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, height1))

    # Save the new image
    new_img.save(os.path.join(dir_save, images1[i]))
