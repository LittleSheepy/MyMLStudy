# 目标检测之选择性搜索-Selective Search
# https://github.com/AlpacaDB/selectivesearch
import skimage
from skimage import segmentation, color,feature
import skimage.segmentation
import numpy as np

class SelectiveSearch:
    def __init__(self, im_orig=None, scale=1.0, sigma=0.8, min_size=50):
        """
        :param im_orig: ndarray. Input img
        :param scale: Higher means larger clusters(聚类) in felzenszwalb segmentation.
        :param sigma: Width of Gaussian kernel for felzenszwalb segmentation. felzenszwalb分割的高斯核宽度。
        :param min_size: Minimum component size for felzenszwalb segmentation. felzenszwalb分割的最小组件大小。
        """
        self.im_orig = im_orig
        self.scale = scale
        self.sigma = sigma
        self.min_size = min_size
    def selective_search(self, im_orig=None):
        im_orig = im_orig or self.im_orig
        assert im_orig.shape[2] == 3, "3ch image is expected"

        # 1. 分割最小区域
        img = self._generate_segments()
        imsize = img.shape[0] * img.shape[1]  # 图片大小

        # 2. 提取区域特征
        R = self._extract_regions(img)      # 提取区域的尺寸，颜色和纹理特征

        # 3. 提取相邻的信息
        # extract neighbouring information
        neighbours = self._extract_neighbours(R)

        # 4. 计算相似度
        # calculate initial similarities
        S = {}
        for (ai, ar), (bi, br) in neighbours:
            S[(ai, bi)] = self._calc_sim(ar, br, imsize)

        # hierarchal search
        while S != {}:
            # get highest similarity
            i, j = sorted(list(S.items()), key=lambda a:a[1], reverse=True)[0][0]
            # merge corresponding regions  合并相应的区域
            t = max(R.keys()) + 1.0
            R[t] = self._merge_regions(R[i], R[j])

            # mark similarities for regions to be removed
            key_to_delete = []
            for k, v in S.items():
                if (i in k) or (j in k):
                    key_to_delete.append(k)
            # remove old similarities of related regions
            for k in key_to_delete:
                del S[k]
            # calculate similarity set with the new region
            for n in set(np.hstack(key_to_delete)):
                if n not in (i, j):
                    S[(t, n)] = self._calc_sim(R[t], R[n], imsize)
        regions = []
        for k, r in R.items():
            regions.append({
                'rect': (
                    r['min_x'], r['min_y'],
                    r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
                'size': r['size'],
                'labels': r['labels']
            })

        return img, regions


    # 分割最小区域
    def _generate_segments(self):
        """
        segment smallest regions by the algorithm of Felzenswalb and Huttenlocher
        通过Felzenswalb和Huttenlocher算法分割最小区域
        :return: region lable is stored in the 4th value of each pixel [r, g, b, region]
        """
        img_float = skimage.util.img_as_float(self.im_orig)  # 图片像素转为0到1的浮点型
        img_msk = skimage.segmentation.felzenszwalb(img_float, self.scale, self.sigma, self.min_size)
        img_msk = np.expand_dims(img_msk, axis=len(img_msk.shape))
        return np.append(self.im_orig, img_msk, axis=2)

    # 提取区域的尺寸，颜色和纹理特征
    def _extract_regions(self, img):
        R = {}

        # get hsv img
        hsv = skimage.color.rgb2hsv(self.im_orig)

        # step 1:count pixel positions 计算像素位置
        for y, pixel in enumerate(img):
            for x, (r,g,b,l) in enumerate(pixel):
                #initialize a new region
                if l not in R:
                    R[l] = {
                        "min_x" :0xffff, "min_y":0xffff,
                        "max_x" :0,      "max_y":0,
                        "labels":[l]
                    }
                # bounding box 更新各个区域边框
                if R[l]["min_x"] > x:
                    R[l]["min_x"] = x
                if R[l]["min_y"] > y:
                    R[l]["min_y"] = y
                if R[l]["max_x"] < x:
                    R[l]["max_x"] = x
                if R[l]["max_y"] < y:
                    R[l]["max_y"] = y

        # step 2: calculate texture gradient
        tex_grad = self._calc_texture_gradient(img)

        # step 3: calculate colour histogram of each region   计算每个区域的颜色直方图
        for k, v in R.items():
            region = img[:,:,3] == k
            masked_pixels = hsv[region]
            R[k]["size"] = len(masked_pixels/4)
            R[k]["hist_c"] = self._calc_colour_hist(masked_pixels)
            # texture histogram
            R[k]["hist_t"] = self._calc_texture_hist(tex_grad[region])
        return R

    # 找邻居 -- 通过计算每个区域与其余的所有区域是否有相交，来判断是不是邻居
    def _extract_neighbours(self, regions):
        def intersect(a, b):
            for x in ["min_x", "max_x"]:
                for y in ["min_y", "max_y"]:
                    if a["min_x"] < b[x] < a["max_x"] and a["min_y"] < b[y] < a["max_y"]:
                        return True
            return False
        R = list(regions.items())
        neighbours = []
        for cur, a in enumerate(R[:-1]):
            for b in R[cur + 1:]:
                if intersect(a[1], b[1]):
                    neighbours.append((a, b))
        return neighbours

    # 计算相似度
    def _calc_sim(self, r1, r2, imsize):
        # calculate the sum of histogram intersection of colour 计算颜色的直方图交集的和
        sim_colour = sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])

        # calculate the sum of histogram intersection of texture  计算纹理的直方图交集的和
        sim_texture = sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])

        # calculate the size similarity over the image  计算图像的大小相似度
        sim_size = 1.0 - (r1["size"] + r2["size"]) / imsize

        # calculate the fill similarity over the image  计算图像的填充相似度
        bbsize = (
                (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
                * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
        )
        sim_fill = 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


        return sim_colour + sim_texture + sim_size + sim_fill


    def _calc_texture_gradient(self, img):
        """
        calculate texture gradient for entire image. 计算整个图像的纹理梯度
        The original SelectiveSearch algorithm proposed Gaussian derivative for 8 orientations, but we use LBP instead.
        原始的SelectiveSearch算法建议用8个方向的高斯导数，但我们改用LBP。
        LBP 局部二值模式 是一种用来描述图像局部纹理特征的算子.
        https://blog.csdn.net/heli200482128/article/details/79204008 （LBP原理介绍以及算法实现）
        :param img:
        :return: output will be [height(*)][width(*)]
        """
        ret = np.zeros(img.shape)
        for colour_channel in range(3):
            ret[:,:,colour_channel] = skimage.feature.local_binary_pattern(img[:,:,colour_channel], P=8, R=1.0)
        return ret

    @staticmethod
    def _calc_colour_hist(img, BINS=25):
        """
        calculate colour histogram of each region   计算每个区域的颜色直方图
        number of bins is 25 as same as [uijlings_ijcv2013_draft.pdf  (论文 Selective Search for Object Recognition)]
        :return:the size of output histogram will be BINS * COLOUR_CHANNELS(3)
        """
        hist = np.array([])
        for colour_channel in range(3):
            # extracting one colour channel 选取一个颜色通道
            c = img[:,colour_channel]
            # calculate histogram for each colour and join to the result 计算每种颜色的直方图并加入结果
            h = np.histogram(c, BINS, (0.0, 255.0))  #  h[0] 数量    h[1] 范围
            hist = np.hstack([hist, h[0]])
        hist = hist/len(img)
        return hist

    @staticmethod
    def _calc_texture_hist(img, BINS=10):
        """
        calculate texture histogram for each region   计算每个区域的纹理直方图
        calculate the histogram of gradient for each colours   计算每个颜色的梯度直方图
        :return:the size of output histogram will be BINS * ORIENTATIONS * COLOUR_CHANNELS(3)
        """
        hist = np.array([])
        for colour_channel in range(3):
            lbps = img[:, colour_channel]
            h = np.histogram(lbps, BINS, (0.0, 255.0))      #  h[0] 数量    h[1] 范围
            hist = np.hstack([hist, h[0]])
        hist = hist/len(img)
        return hist

    def _merge_regions(self, r1, r2):
        new_size = r1["size"] + r2["size"]
        rt = {
            "min_x": min(r1["min_x"], r2["min_x"]),
            "min_y": min(r1["min_y"], r2["min_y"]),
            "max_x": max(r1["max_x"], r2["max_x"]),
            "max_y": max(r1["max_y"], r2["max_y"]),
            "size": new_size,
            "hist_c": (r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
            "hist_t": (r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
            "labels": r1["labels"] + r2["labels"]
        }
        return rt


if __name__ == '__main__':
    import cv2
    import math
    import sys
    import skimage.io
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    img_path = 'D:\ML_datas/17flowers/jpg/7/image_0562.jpg'
    img = cv2.imread(img_path)
    img_lbl, regions = SelectiveSearch(img, scale=4000, sigma=0.9, min_size=10).selective_search()
    rects = set()
    for r in regions:
        rects.add(r["rect"])
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    img = skimage.io.imread(img_path)
    ax.imshow(img)
    for x, y, w, h in rects:
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()

























