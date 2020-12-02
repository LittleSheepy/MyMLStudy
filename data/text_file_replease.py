import os





def main():
    src_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml_txt/"
    out_dir = "D:/01sheepy/01work/01baojie_ocr/pp/xml_txt_new/"
    for text_file in os.listdir(src_dir):
        f = open(os.path.join(src_dir, text_file),"r")
        f_out = open(os.path.join(out_dir, text_file),"a")
        text = f.read()
        text = text.replace("####",text_file[:-4])
        f_out.write(text)
        f.close()
        f_out.close()



if __name__ == '__main__':
    main()