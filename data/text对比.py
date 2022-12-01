import os



def txt2txt_compare():
    txt1_path = r"D:\01sheepy\01work\06ningbo\imagessssssssssssss\imgs/name.txt"
    txt2_path = r"D:\01sheepy\01work\06ningbo\imagessssssssssssss\imgs/result.txt"
    txt3_path = r"D:\01sheepy\01work\06ningbo\imagessssssssssssss\imgs/diff.txt"
    txt1file = open(txt1_path)
    txt2file = open(txt2_path)
    txt3file = open(txt3_path,mode='a')
    txt1lines = txt1file.readlines()
    txt2lines = txt2file.readlines()
    set_rec = set(txt1lines).difference(set(txt2lines))
    for line in set_rec:
        txt3file.write(line)
    txt1file.close()
    txt2file.close()
    txt3file.close()

if __name__ == '__main__':
    txt2txt_compare()