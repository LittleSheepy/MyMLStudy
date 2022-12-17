import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import zipfile
import cv2
import random

class Celeba:
    def __init__(self, path, path_ann, path_bbox):
        self._process_imgs(path)
        self._process_anns(path_ann)
        self._process_bboxs(path_bbox)

        self.pos = np.random.randint(0, self.num_examples)
        self.persons = max(self.ids) + 1
        print(f"Persons = {self.persons}")
        assert self.persons == len(set(self.ids))

    def _process_bboxs(self, path_bbox):
        with open(path_bbox) as file:
            lines = file.readlines()
            del lines[0]
            del lines[0]
            self.bboxes = []
            for line in lines:
                locs = []
                for loc in line.split(" "):
                    if len(loc) > 0:
                        locs.append(loc)
                del locs[0]
                locs = [int(e) for e in locs]
                self.bboxes.append(locs)

    def _process_anns(self, path_ann):
        with open(path_ann) as file:
            lines = file.readlines()
        self.ids = []
        for line in lines:
            id = int(line[line.find(" ") + 1:]) - 1
            self.ids.append(id)

    def _process_imgs(self, path):
        self.filenames = []
        self.zf = zipfile.ZipFile(path)
        for info in self.zf.filelist:
            if info.is_dir(): continue
            self.filenames.append(info.filename)
            if len(self.filenames) % 10000 == 0:
                print(f"read {len(self.filenames)} imgs")

        #random.shuffle(self.filenames)
        print(f"read {len(self.filenames)} from {path} success")

    @property
    def num_examples(self):
        return len(self.filenames)

    def next_batch(self, batch_size):
        next = self.pos + batch_size
        num = self.num_examples
        if next < num:
            result = self.filenames[self.pos:next], self.ids[self.pos:next], self.bboxes[self.pos:next]
        else:
            result = self.filenames[self.pos:num], self.ids[self.pos:next], self.bboxes[self.pos:next]
            next -= num
            result += self.filenames[:next], self.ids[self.pos:next], self.bboxes[self.pos:next]
        self.pos = next

        res = []
        for filename in result[0]:
            img = self.zf.read(filename)
            img = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img, 1)
            res.append(img)
        return res, result[1]
    def close(self):
        self.zf.close()
    def __enter__(self):
        pass
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

if __name__ == '__main__':
    path = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Img/img_align_celeba.zip"
    path_anno = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Anno/identity_CelebA.txt"
    path_bbox = "D:/BaiduYunDownload/15_Ce.leb.Faces Att.ribu.tes Da.ta.set (CelebA)/Anno/list_bbox_celeba.txt"
    ce = Celeba(path, path_anno, path_bbox)
    with ce:
        ds = ce.next_batch(10)
        cv2.imshow("a",ds[0])
        cv2.waitKey()


















