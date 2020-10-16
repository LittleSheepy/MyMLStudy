# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
import random
import colorsys
import cv2
from PIL import Image, ImageDraw, ImageFont

from lib.config.Config import config as config
print("import utils")
def test_utils():
    print("test_utils")

def get_shape(x, rank=None):
    """
    Returns the dimensions of a Tensor as list of integers or scale tensors.
    Args:
      x: N-d Tensor;
      rank: Rank of the Tensor. If None, will try to guess it.
    Returns:
      A list of `[d1, d2, ..., dN]` corresponding to the dimensions of the
        input tensor.  Dimensions that are statically known are python integers,
        otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape()
        if rank is None:
            static_shape = static_shape.as_list()
            rank = len(static_shape)
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]
# 激活函数
def leaky_relu(x):  # leaky relu激活函数，leaky_relu激活函数一般用在比较深层次神经网络中
    return tf.maximum(0.1 * x, x)
    # return tf.nn.leaky_relu(x,alpha=0.1,name='leaky_relu') # 或者tf.maximum(0.1*x,x)

conv_num = 1
# Conv+BN：yolo2中每个卷积层后面都有一个BN层， batch normalization正是yolov2比yolov1多的一个东西，可以提升mAP大约2%
def conv2d(x, filters_num, filters_size, pad_size=0, stride=1, batch_normalize=True, activation=leaky_relu,
           use_bias=False, name=None):
    global conv_num
    if name is None or name == "conv%d"%conv_num:
        name = "conv%d"%conv_num
        conv_num += 1
    # padding，注意: 不用padding="SAME",否则可能会导致坐标计算错误，用自己定义的填充方式
    if pad_size > 0:
        x = tf.pad(x, [[0, 0], [pad_size, pad_size], [pad_size, pad_size],
                       [0, 0]])  # 这里使用tensorflow中的pad进行填充,主要填充第2和第3维度，填充的目的是使得经过卷积运算之后，特征图的大小不会发生变化

    out = tf.layers.conv2d(x, filters=filters_num, kernel_size=filters_size, strides=stride, padding='VALID',
                           activation=None, use_bias=use_bias, name=name)
    # BN应该在卷积层conv和激活函数activation之间,(后面有BN层的conv就不用偏置bias，并激活函数activation在后)
    if batch_normalize:  # 卷积层的输出，先经过batch_normalization
        out = tf.layers.batch_normalization(out, axis=-1, momentum=0.9, training=config.training, name=name + '_bn')
    if activation:  # 经过batch_normalization处理之后的网络输出输入到激活函数
        out = activation(out)
    return out  # 返回网络的输出

# 残差网络
def residual_block(input_data, filter_num1, filter_num2, name):
    short_cut = input_data
    with tf.variable_scope(name):
        input_data = conv2d(input_data, filter_num1, 1, 0)
        input_data = conv2d(input_data, filter_num2, 3, 1)
        residual_output = input_data + short_cut
    return residual_output

# 合并不同层数据
def route(name, previous_output, current_output):
    with tf.variable_scope(name):
        output = tf.concat([current_output, previous_output], axis=-1)
    return output

# 上采样
def upsample(input_data, name, method="deconv"):
    assert method in ["resize", "deconv"]
    if method == "resize":
        with tf.variable_scope(name):
            input_shape = tf.shape(input_data)
            output = tf.image.resize_nearest_neighbor(input_data, (input_shape[1] * 2, input_shape[2] * 2))
    else:
        # replace resize_nearest_neighbor with conv2d_transpose To support TensorRT optimization
        numm_filter = input_data.shape.as_list()[-1]
        output = tf.layers.conv2d_transpose(input_data, numm_filter, kernel_size=2, padding='same',
                                            strides=(2,2), kernel_initializer=tf.random_normal_initializer())
    return output

# max_pool
def maxpool(x,size=2,stride=2,name='maxpool'):      #maxpool，最大池化层
    return tf.layers.max_pooling2d(x,pool_size=size,strides=stride,name=name)

# reorg layer(带passthrough的重组层)，主要是利用到Fine-Grained Feature（细粒度特征用于检测微小物体）
def reorg(x,stride):
    return tf.space_to_depth(x,block_size=stride)   #返回一个与input具有相同的类型的Tensor
    # return tf.extract_image_patches(x,ksizes=[1,stride,stride,1],strides=[1,stride,stride,1],rates=[1,1,1,1],padding='VALID')



####################        图像处理函数           ########################################################################

# 【1】图像预处理(pre process前期处理)
def preprocess_image(image,target_size=(416,416), process="padding"):
    # 复制原图像
    #image_cp = np.copy(image).astype(np.float32)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
    if process == "padding":
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))
        image = np.full(shape=[ih, iw, 3], fill_value=128.0)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image[dh:nh + dh, dw:nw + dw, :] = image_resized
    elif process == "resize":
        image = cv2.resize(image, target_size)
    # normalize归一化
    image /= 255.0

    # 增加一个维度在第0维——batch_size
    image = np.expand_dims(image,axis=0)
    return image

def scores_process(obj_probs, class_probs, threshold=0.5):
    obj_probs = np.reshape(obj_probs,[-1])                          #13*13*5=845
    class_probs = np.reshape(class_probs,[len(obj_probs),-1])       #[13*13*5,80]  [845, 20]
    class_max_index = np.argmax(class_probs,axis=1)                 # 得到max类别概率对应的维度 [845, 1]
    class_probs = class_probs[np.arange(len(obj_probs)),class_max_index]  # [845]
    scores = obj_probs * class_probs   #Confidence置信度*max类别概率=类别置信度scores # [845]

    # 类别置信度scores>threshold的边界框bboxes留下
    score_mask = scores > threshold
    return score_mask, scores, class_max_index

# 【2】筛选解码后的回归边界框——NMS(post process后期处理)
def postprocess(bboxes,obj_probs,class_probs,image_shape=(416,416),threshold=0.3):
    bboxes = np.reshape(bboxes,[-1,4])
    score_mask, scores, class_max_index = scores_process(obj_probs, class_probs, threshold)
    class_max_index = class_max_index[score_mask]   # [5]
    scores = scores[score_mask]    #一一对应，得到保留下来的框的类别置信度
    bboxes = bboxes[score_mask]    #一一对应，得到保留下来的框(左上-右下)

    # 将所有box还原成图片中真实的位置
    bboxes[:,0:1] *= float(image_shape[1]) # xmin*width
    bboxes[:,1:2] *= float(image_shape[0]) # ymin*height
    bboxes[:,2:3] *= float(image_shape[1]) # xmax*width
    bboxes[:,3:4] *= float(image_shape[0]) # ymax*height
    bboxes = bboxes.astype(np.int32)    #转变为int类型

    # cut the box:将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    bboxes = np.clip(bboxes, [0,0,0,0], [image_shape[1]-1, image_shape[0]-1, image_shape[1]-1, image_shape[0]-1])

    # 排序top_k(默认为400)
    class_max_index,scores,bboxes = bboxes_sort(class_max_index,scores,bboxes)  #只保留类别置信度为top k的k个
    # NMS
    class_max_index,scores,bboxes = bboxes_nms(class_max_index,scores,bboxes)  #关键的一步，非极大值抑制
    return bboxes,scores,class_max_index   #返回保存下来的框

def generate_colors(class_names):  #为每个类别显示不同的颜色，哈哈，颜值
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# 【3】绘制筛选后的边界框
def draw_detection(im, bboxes, scores, cls_inds, labels, colors, thr=0.3): #传入的参数分别为原始图片，最终得到的框，类别置信度，类别索引，类别库，各个类别的颜色，thr为类别置信度阈值
    # draw image
    imgcv = np.copy(im)
    h, w, _ = imgcv.shape    #得到图片的高度和宽度
    for i, box in enumerate(bboxes):
        if scores[i] < thr:    #我感觉多此一举，因为前面已经过滤过一次socres了
            continue
        cls_indx = cls_inds[i]

        thick = int((h + w) / 1000)
        #cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),(255, 114, 0), thick)
        cv2.rectangle(imgcv,(box[0], box[1]), (box[2], box[3]),(74, 120, 237),thick)  #显示框
        mess = '%s: %.3f' % (labels[cls_indx], scores[i])
        if box[1] < 20:
            text_loc = (box[0] + 2, box[1] + 15)
        else:
            text_loc = (box[0], box[1] - 10)

        #cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, colors[cls_indx], thick//3)
        cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3*h, (184, 166, 51), thick)  #显示类别和类别置信度
    return imgcv  #返回图片

# (2)按类别置信度scores降序，对边界框进行排序并仅保留top_k
def bboxes_sort(classes,scores,bboxes,top_k=400):
    index = np.argsort(-scores)
    classes = classes[index][:top_k]  #0~79
    scores = scores[index][:top_k]
    bboxes = bboxes[index][:top_k]
    return classes,scores,bboxes   #这三个变量分别表示类别（0，79）针对于coco数据集，scores表示这个框的类别置信度，bboxes表示存储着框的四个值（左上-右下）

def postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold=0.5):
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    score_mask, scores, class_max_index = scores_process(pred_conf, pred_prob, score_threshold)
    class_max_index = class_max_index[score_mask]   # [5]
    scores = scores[score_mask]    #一一对应，得到保留下来的框的类别置信度
    pred_xywh = pred_xywh[score_mask]    #一一对应，得到保留下来的框(左上-右下)

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    pred_coor = np.clip(pred_coor, [0,0,0,0], [org_w-1, org_h-1, org_w-1, org_h-1])
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    valid_scale=[0, np.inf]
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
    coors, scores, classes = pred_coor[scale_mask], scores[scale_mask], class_max_index[scale_mask]
    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)







def postprocess_boxes1(pred_bbox, org_img_shape, input_size, score_threshold=0.5):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes those are out of range 剪掉一些超出范围的盒子
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes 丢弃一些无效的boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores 丢弃一些分数低的
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

# 计算两个box的IOU
def bboxes_iou(bboxes1,bboxes2):   #计算IOU的值
    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)

    # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
    int_h = np.maximum(int_ymax-int_ymin,0.)
    int_w = np.maximum(int_xmax-int_xmin,0.)

    # 计算IOU
    int_vol = int_h * int_w # 交集面积
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1]) # bboxes1面积
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1]) # bboxes2面积
    IOU = int_vol / (vol1 + vol2 - int_vol) # IOU=交集/并集
    return IOU

def nms(bboxes, iou_threshold=0.45, sigma=0.5, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = np.copy(bboxes[cls_mask])

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.25
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def draw_bbox(image, bboxes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    classes = config.class_names
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image



if __name__ == '__main__':
    pass






