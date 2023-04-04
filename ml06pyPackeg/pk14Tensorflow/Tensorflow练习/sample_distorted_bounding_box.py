import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pprint
import cv2



def bboxes_resize(bbox_ref, bboxes, name=None):
    """Resize bounding boxes based on a reference bounding box,
    assuming that the latter is [0, 0, 1, 1] after transform. Useful for
    updating a collection of boxes after cropping an image.
    """
    # Bboxes is dictionary.
    if isinstance(bboxes, dict):
        with tf.name_scope(name, 'bboxes_resize_dict'):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes

    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes

img_path = "D:\ML_datas\VOCdevkit\VOC2007/train\JPEGImages/000005.jpg"
image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
with tf.Session() as Sess:
    ima_data = tf.image.decode_png(image_raw_data)
    image_float = tf.image.convert_image_dtype(ima_data, dtype=tf.float32)
    bbox = [0.5, 0.5, 0.6, 0.6]
    boxes = tf.constant([[bbox]])

    original_image = tf.expand_dims(image_float, 0)
    image_with_box = tf.image.draw_bounding_boxes(original_image, boxes=boxes)
    plt.figure(1)
    plt.imshow(image_with_box.eval().reshape([375, 500, 3]))
    plt.show()

    for i in range(1):
        bbox_begin, bbox_size, bbox_2 = tf.image.sample_distorted_bounding_box(tf.shape(image_float),
                                                                               bounding_boxes=boxes,
                                                                               min_object_covered=0.3,
                                                                               aspect_ratio_range=[1, 1])
        distort_image = tf.slice(image_float, bbox_begin, bbox_size)
        distort_image.set_shape([None, None, 3])
        bbox_00 = bbox_2[0, 0]
        bboxes_resize = bboxes_resize(bbox_00, tf.squeeze(boxes,axis=0))
        bboxes_resize_e = tf.expand_dims(bboxes_resize, dim=0)

        distort_image_e = tf.expand_dims(distort_image, dim=0)
        image_with_box = tf.image.draw_bounding_boxes(distort_image_e, boxes=bboxes_resize_e)


        image_with_box1 = tf.image.draw_bounding_boxes(original_image, boxes=bbox_2)

        #image_with_box = tf.squeeze(image_with_box,axis=0)
        bb, bbb, bbz,distort_image_v,image_with_box_v,image_with_box1_v = Sess.run([bbox_2, bbox_begin, bbox_size,distort_image,image_with_box,image_with_box1])
        plt.figure(1)
        plt.imshow(distort_image_v)
        plt.show()
        plt.figure(1)
        plt.imshow(image_with_box_v[0])
        plt.show()
        plt.figure(1)
        plt.imshow(image_with_box1_v[0])
        plt.show()
        print(distort_image_v.shape)



        print('bbox: ', bb)
        print('bbox_begin: ', bbb)
        print('bbox_size: ', bbz)

        bbox_2 = bbox_2[0, 0]


