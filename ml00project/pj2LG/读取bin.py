import struct

# Define the structure of CDefect in Python
class CDefect:
    def __init__(self, member1, member2, member3, member4):
        self.member1 = member1
        self.member2 = member2
        self.member3 = member3
        self.member4 = member4


root_path = r"D:\04DataSets\ningjingLG\AI_para/"
bin_path = root_path + "1_0.bin"
# Open the binary file for reading
with open(bin_path, "rb") as f:
    read_defects = []
    while True:
        # Read the binary data for one CDefect object
        binary_data = f.read(struct.calcsize("iiii"))
        if not binary_data:
            break
        # Unpack the binary data into a CDefect object
        temp_defect = CDefect(*struct.unpack("iiii", binary_data))
        read_defects.append(temp_defect)