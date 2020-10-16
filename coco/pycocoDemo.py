#!/usr/bin/env python
# coding: utf-8

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='D:/ML_datas/coco/coco2014'
dataType='train2014'
dataTest = "test2014"
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)  # captions
captions='{}/annotations/captions_{}.json'.format(dataDir,dataType)  # captions
person_keypoints_path='{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)  # captions
image_info_path='{}/annotations/image_info_{}.json'.format(dataDir,dataTest)  # captions

# initialize COCO api for instance annotations
# coco=COCO(annFile)
# capt = COCO(captions)
# person_keypoints = COCO(person_keypoints_path)
image_info = COCO(image_info_path)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person'])      # 'person','dog','skateboard'
imgIds = coco.getImgIds(catIds=catIds)
#imgIds = coco.getImgIds(imgIds = [142589])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# load and display image
# I = io.imread('%s/images/%s/%s'%(dataDir,dataType,img['file_name']))
# use url to load image
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()

# load and display instance annotations
plt.imshow(I)
plt.axis('off')
annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()

# initialize COCO api for person keypoints annotations
annFile = '{}/annotations/person_keypoints_{}.json'.format(dataDir,dataType)
coco_kps=COCO(annFile)

# load and display keypoints annotations
plt.imshow(I);
plt.axis('off')
ax = plt.gca()
annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
anns = coco_kps.loadAnns(annIds)
coco_kps.showAnns(anns)
plt.show()

# initialize COCO api for caption annotations
annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

# load and display caption annotations
annIds = coco_caps.getAnnIds(imgIds=img['id'])
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I)
plt.axis('off')
plt.show()



