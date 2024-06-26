import os
from PIL import Image


def find_matching_images(directory):
    # Dictionary to hold groups of matching images
    image_groups = {}

    # Iterate through all files in the directory
    for filename in os.listdir(directory):
        # Split the filename into name and extension
        name, extension = os.path.splitext(filename)

        # Ignore non-image files
        if extension.lower() not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
            continue

        # Extract the base name without the last two characters (assuming format XXX1_1)
        base_name = name[:-3]

        # Add the filename to the corresponding group in the dictionary
        if base_name in image_groups:
            image_groups[base_name].append(filename)
        else:
            image_groups[base_name] = [filename]

    # Filter out groups that don't have exactly 4 images
    matching_groups = {k: v for k, v in image_groups.items() if len(v) == 4}

    return matching_groups


def stitch_images(directory, directory_save, image_group):
    images = [Image.open(os.path.join(directory, filename)) for filename in image_group]

    # Assuming all images are of the same size
    width, height = images[0].size

    # Create a new image with a width and height that's twice that of the individual images
    new_image = Image.new('RGB', (width * 2, height * 2))

    # Paste the individual images into the new image
    new_image.paste(images[0], (0, 0))
    new_image.paste(images[1], (0, height))
    new_image.paste(images[2], (width, 0))
    new_image.paste(images[3], (width, height))

    # Save the new image
    new_image.save(os.path.join(directory_save, image_group[0][:-3] + '.jpg'))


def get_directory_list(directory):
    directory_list = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            directory_list.append(item)

    return directory_list

def main():
    directory_list = get_directory_list(root_dir)
    for directory_name in directory_list:

        directory = os.path.join(root_dir, directory_name)
        directory_save = os.path.join(root_dir, directory_name+"_big")
        if not os.path.exists(directory_save):
            os.mkdir(directory_save)
        matching_groups = find_matching_images(directory)

        for group in matching_groups.values():
            stitch_images(directory, directory_save, group)


if __name__ == '__main__':
    # root_dir = r"E:\点检和测试\点检图片\整个托盘/"
    root_dir = r"./"
    main()
