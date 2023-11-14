import os

def delete_empty_dirs(directory):
    if not os.listdir(directory):
        os.rmdir(directory)
    else:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                delete_empty_dirs(subdir_path)

delete_empty_dirs(r'C:\Users\KADO\AppData\Local\Temp')