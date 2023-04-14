import math
import numpy as np

from lib.config.Config import config as config


def generate_anchors_by_size_ratios():
    anchors = []
    for i, [sizes, ratios] in enumerate(zip(config.anchor_sizes, config.anchor_ratios)):
        anchors_l = []
        anchors_l.append([sizes[0], sizes[0]])
        if len(sizes) > 1:
            w = math.sqrt(sizes[0] * sizes[1])
            h = math.sqrt(sizes[0] * sizes[1])
            anchors_l.append([w, h])
        for ratio in ratios:
            h = sizes[0] / math.sqrt(ratio)
            w = sizes[0] * math.sqrt(ratio)
            anchors_l.append([w, h])
        anchors.append(np.array(anchors_l))
    return anchors


def generate_grids(steps=None):
    grids = []
    feature_shapes = config.feature_shapes
    for i, feat_shape in enumerate(feature_shapes):
        yx = np.mgrid[0:feat_shape, 0:feat_shape] * 1.0
        feat_shape = np.array([feat_shape,feat_shape]).reshape([2, 1, 1])
        if steps:
            c_yx = (yx + 0.5) * steps[i]
        else:
            c_yx = (yx + 0.5) / feat_shape
        c_yx = np.transpose(c_yx,[1,2,0])
        c_xy = c_yx[:,:,[1,0]]
        # c_xy = c_xy[:,:,:,np.newaxis]
        grids.append(c_xy)
    return grids

def bboxes_encode(glabels, gbboxes, anchors, grids, feat_shape=[3,3], img_shape=[300,300]):
    # anchors
    anchors = np.array(anchors)
    anchors = anchors / img_shape[0]
    hs = anchors[:, 1]
    ws = anchors[:, 0]
    href = hs
    wref = ws
    vol_anchors = hs * ws
    # grids 38 38 2
    grids = np.array(grids)
    if grids.max() > 2:
        grids = grids/img_shape
    cy, cx = grids[:,:,1,np.newaxis],grids[:,:,0,np.newaxis]
    yref, xref = grids[:,:,1,np.newaxis],grids[:,:,0,np.newaxis]

    ymin = cy - hs / 2
    xmin = cx - hs / 2
    ymax = cy + hs / 2
    xmax = cx + ws / 2

    shape = (feat_shape[0], feat_shape[1], len(anchors))  # (38, 38, 4)
    feat_labels = np.zeros(shape, dtype=np.int64)
    feat_scores = np.zeros(shape, dtype=np.float32)
    feat_ymin = np.zeros(shape, dtype=np.float32)
    feat_xmin = np.zeros(shape, dtype=np.float32)
    feat_ymax = np.ones(shape, dtype=np.float32)
    feat_xmax = np.ones(shape, dtype=np.float32)
    for i,[label, bbox]  in enumerate(zip(glabels, gbboxes)):
        int_ymin = np.maximum(ymin, bbox[0])            # (38, 38, 4)
        int_xmin = np.maximum(xmin, bbox[1])
        int_ymax = np.minimum(ymax, bbox[2])
        int_xmax = np.minimum(xmax, bbox[3])
        h = np.maximum(int_ymax - int_ymin, 0.)
        w = np.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w
        vol_box = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        union_vol = vol_anchors - inter_vol + vol_box
        ious = inter_vol/union_vol          # (38, 38, 4)

        mask = ious > feat_scores
        imask = mask.astype(np.int64)
        fmask = mask.astype(np.float64)
        # Update values using mask.
        feat_labels = imask * label + (1 - imask) * feat_labels
        feat_scores = np.where(mask, ious, feat_scores)

        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax

    # Transform to center / size.
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin
    # Encode features.
    prior_scaling = [0.1, 0.1, 0.2, 0.2]
    feat_cy1 = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx1 = (feat_cx - xref) / wref / prior_scaling[1]
    feat_h1 = np.log(feat_h / href) / prior_scaling[2]
    feat_w1 = np.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.
    feat_localizations = np.stack([feat_cx1, feat_cy1, feat_w1, feat_h1], axis=-1)
    return feat_labels, feat_localizations, feat_scores

def bboxes_decode(feat_locs, anchors, grids, feat_shape=[3,3], img_shape=[300,300], prior_scaling=[0.1, 0.1, 0.2, 0.2]):
    anchors = np.array(anchors)
    if anchors.max() > 2:
        anchors /= img_shape
    # grid
    if grids.max() > 2:
        grids = grids / img_shape

    # Compute center, height and width
    cxy = feat_locs[:, :, :, 0:2] * anchors * prior_scaling[:2] + grids[:,:,np.newaxis,:]
    wh = anchors * np.exp(feat_locs[:, :, :, 2:] * prior_scaling[2:])
    minxy = cxy - wh/2
    maxxy = cxy + wh/2
    bboxes = np.concatenate([minxy, maxxy], axis=-1)
    return bboxes

def detected_bboxes():
    pass

##############  outputs process
def scores_select(bboxes, scores, classes, topK=100, threshold=0.5):
    mask = scores > threshold
    classes = classes[mask]
    scores = scores[mask]
    bboxes = bboxes[mask]
    index = np.argsort(-scores)
    classes = classes[index][:topK]
    scores = scores[index][:topK]
    bboxes = bboxes[index][:topK]
    return bboxes, scores, classes

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

# NMS，或者用tf.image.non_max_suppression(boxes, scores,self.max_output_size, self.iou_threshold)
def bboxes_nms(classes, scores, bboxes, nms_threshold=0.5, sigma=0.5, method='nms'):   #iou阈值为0.5， 这些值已经按照从高到底进行排序
    assert method in ['nms', 'soft-nms'], "method must in [nms, soft-nms]"
    scores_org = np.copy(scores)
    keep_bboxes = np.ones(scores.shape, dtype=np.bool)
    for i in range(scores.size-1):
        if keep_bboxes[i]:
            # Computer overlap with bboxes which are following.
            overlap = bboxes_iou(bboxes[i], bboxes[(i+1):])
            if method == "nms":
                # Overlap threshold for keeping + checking part of the same class 逻辑或，IOU没有超过0.5或者是不同的类则保存下来
                keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
            else :
                mask = classes[(i+1):] == classes[i]
                soft_score = (np.exp(-(1.0 * overlap ** 2 / sigma)))*scores[(i+1):]
                scores[(i+1):][mask] = soft_score[mask]
                keep_overlap = scores[(i+1):] > 0.25
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)   #逻辑与

    idxes = np.where(keep_bboxes)
    return classes[idxes], scores_org[idxes], bboxes[idxes]   #返回保存下来的框


"""
[ 1          6            6          1         1              6          1       6          1         1          11          11         11          11             6]
[0.9933234  0.98938304 0.9832955  0.9623017  0.95860755 0.9280947, 0.92020315 0.91179895 0.89027613 0.8852529  0.8082116  0.76939136, 0.71820825 0.71794707 0.6044171 ]
           [0.99726333 0.99748344 0.22783128 0.1567128  0.99735857 0.19652309, 0.99694812 0.20369481 0.22510864 0.87923857 0.87317412 0.87729811, 0.87811018 0.99730695]   ww
           [0.98667543 0.98082097 0.21924242 0.15022608 0.92564319 0.18084117, 0.90901626 0.18134463 0.19927807 0.71061083 0.67181262 0.63008274, 0.63043662 0.60278936]  soft_score 0.5
           [0.98487447 0.97917475 0.08178371 0.04366649 0.92401246 0.06112793, 0.90716585 0.06278049 0.07374303 0.65218326 0.61373666 0.57742517, 0.57810595 0.60170664]  soft_score0.3
[0.9933234  0.98938304 0.9832955  0.2805363  0.20459479 0.9280947, 0.23717107 0.91179895 0.23641482 0.25550193 0.8082116  0.76939136, 0.71820825 0.71794707 0.6044171 ]
[0.9933234  0.98938304 0.9832955  0.21924242 0.15022607 0.9280947, 0.18084116 0.91179895 0.18134463 0.19927807 0.8082116  0.76939136, 0.71820825 0.71794707 0.6044171 ]   0.5
[0.9933234  0.98938304 0.9832955  0.08178371 0.04366649 0.9280947, 0.06112793 0.91179895 0.06278048 0.07374302 0.8082116  0.76939136, 0.71820825 0.71794707 0.6044171 ]   0.3
[0.9933234  0.98938304 0.9832955  0.38178563 0.30101016 0.9280947, 0.33286428 0.91179895 0.3293345  0.34858802 0.8082116  0.76939136, 0.71820825 0.71794707 0.6044171 ]  0.8 

"""



if __name__ == '__main__':
    # glabels = np.array([1])
    # gbboxes = np.array([[200, 200, 250, 250]])
    # anchors = generate_anchors_by_size_ratios()
    # grids = generate_grids(config.steps)
    # feat_labels, feat_localizations, feat_scores = bboxes_encode(glabels, gbboxes, anchors[4],grids[4])
    # print(feat_labels)
    # bboxes_decode(feat_localizations, anchors[4],grids[4])
    #

    b=4
