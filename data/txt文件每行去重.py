import os

def txtLineDelRepet():
    lines = []
    with open(dst_file, 'w') as file_new:
        with open(src_file, 'r') as file:
            for line in file:
                if not line in lines:
                    print(line)
                    file_new.write(line)
                    lines.append(line)



if __name__ == '__main__':
    dir_root = r"D:/0/0LG_DATA/052120_052208/"
    src_file = dir_root + r"DM_SZ.txt"
    dst_file = dir_root + r"DM_SZ_new.txt"
    txtLineDelRepet()



