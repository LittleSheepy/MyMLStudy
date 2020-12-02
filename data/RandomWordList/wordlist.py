import os
import cv2
import numpy as np
import random

pic_num_init = 250

def rand_data_list_generate(indir, outdir):
    global pic_num_init
    wordlist = []
    for file in os.listdir(indir):
        print(file)
        wordlist.append(file)
    wordcnt = len(wordlist)
    f_label = open(os.path.join(outdir, "label.txt"), "a")
    for i in range(50):
        #data_cnt = random.randint(2,4)
        data_cnt = 2
        data_num = [random.randint(0,wordcnt-1) for i in range(data_cnt)]
        img = None
        label = ""
        for i in data_num:
            label += wordlist[i][:-4]
            imgtmp = cv2.imread(os.path.join(indir, wordlist[i]))
            height, width = imgtmp.shape[0], imgtmp.shape[1]
            radio = height/256
            widthnew = width/radio
            imgtmp = cv2.resize(imgtmp, (int(widthnew), 256))
            # imgtmp = cv2.imread('D:/01sheepy/work/baojie_ocr/words/olay/3.jpg')
            if img is None:
                img = imgtmp
                continue
            img = np.hstack([img, imgtmp])
        cv2.imwrite(outdir+"/g"+str(pic_num_init)+".jpg", img)
        f_label.write("train/" + "g"+str(pic_num_init)+".jpg" + '\t' + label + '\n')
        pic_num_init += 1
    f_label.close()


    # while True:
    #     cv2.imshow("result", img)
    #     cv2.waitKey(0)
    #cv2.imshow("22", img)




def main():
    in_dir = "F:/tongli/words/"
    out_dir = "F:tongli/words_list/"
    # rand_data_list_generate(in_dir, out_dir)
    for wordsdir in os.listdir(in_dir):
        in_wordsdir = os.path.join(in_dir, wordsdir)
        rand_data_list_generate(in_wordsdir, out_dir)



if __name__ == '__main__':
    main()