#coding=utf8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import cv2
import os,sys
import scipy.ndimage
import time
import scipy

def corner_nms(corner,kernal=3):
	out = corner.copy()
	row_s = int(kernal/2)
	row_e = out.shape[0] - int(kernal/2)
	col_s,col_e = int(kernal/2),out.shape[1] - int(kernal/2)
	for r in range(row_s,row_e):
		for c in range(col_s,col_e):
			if corner[r,c]==0: #不是可能的角点
				continue
			zone = corner[r-int(kernal/2):r+int(kernal/2)+1,c-int(kernal/2):c+int(kernal/2)+1]
			index = corner[r,c]<zone
			(x,y) = np.where(index==True)
			if len(x)>0 : #说明corner[r,c]不是最大，直接归零将其抑制
				out[r,c] = 0
			else:
				out[r,c] = 255
	return out



def harris_corner_detect(img_src,block_size=2,aperture_size=3,k=0.04,borderType=cv2.BORDER_DEFAULT):
	if img_src.dtype!=np.uint8:
		raise ("input image shoud be uint8 type")
	R_arr = np.zeros(img_src.shape,dtype=np.float32)#用来存储角点响应值
	img = img_src.astype(np.float32)
	scale = 1.0/( (aperture_size-1)*2*block_size*255 )#参考opencv实现源码，在sobel算子时乘以这个系数
	#借助sobel算子，计算x、y方向上的偏导数
	Ix = cv2.Sobel(img,-1,dx=1,dy=0,ksize=aperture_size,scale=scale,borderType=borderType)
	Iy = cv2.Sobel(img,-1,dx=0,dy=1,ksize=aperture_size,scale=scale,borderType=borderType)
	Ixx = Ix**2
	Iyy = Iy**2
	Ixy = Ix*Iy
	#借助boxFilter函数，以block_size为窗口大小，对窗口内的数值求和，且不归一化
	f_xx = cv2.boxFilter(Ixx,ddepth=-1,ksize=(block_size,block_size) ,anchor =(-1,-1),normalize=False, borderType=borderType)
	f_yy = cv2.boxFilter(Iyy,ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	f_xy = cv2.boxFilter(Ixy, ddepth=-1,ksize=(block_size,block_size),anchor =(-1,-1),normalize=False,borderType=borderType)
	# 也可以尝试手动求和
	radius = int((block_size - 1) / 2)  # 考虑blocksize为偶数的情况，奇数时，前、后的数量一样；为偶数时，后比前多一个
	N_pre = radius
	N_post = block_size - N_pre - 1
	row_s, col_s = N_pre, N_pre
	row_e, col_e = img.shape[0] - N_post, img.shape[1] - N_post
	#开始计算每一个坐标下的响应值
	for r in range(row_s,row_e):
		for c in range(col_s,col_e):
			#手动对窗口内的数值求和
			#sum_xx = Ixx[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			#sum_yy = Iyy[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			#sum_xy = Ixy[r-N_pre:r+N_post+1,c-N_pre:c+N_post+1].sum()
			#或者直接使用boxFilter的结果
			sum_xx = f_xx[r,c]
			sum_yy = f_yy[r, c]
			sum_xy = f_xy[r, c]
			#根据行列式和迹求响应值
			det = (sum_xx * sum_yy) - (sum_xy ** 2)
			trace = sum_xx + sum_yy
			res = det - k * (trace ** 2)
			# 或者用如下方式求行列式和迹
			#M = np.array([[sum_xx,sum_xy],[sum_xy,sum_yy]])
			#res = np.linalg.det(M) - (k *  (np.trace(M))**2 )
			R_arr[r,c] = res
	return R_arr



if __name__ == '__main__':
	image_path = r'D:\04DataSets\04\box.jpg'
	img_src = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
	#source code
	res_arr = harris_corner_detect(img_src,block_size=2,aperture_size=3,k=0.04)
	max_v = np.max(res_arr)
	res_arr[res_arr<0.01*max_v]=0
	img_show = img_src.copy()
	if(len(img_show.shape)==2):
		img_show = cv2.cvtColor(img_show,cv2.COLOR_GRAY2BGR)
	img_show[res_arr!=0] = (255,0,0)
	print(len(np.where(res_arr!=0)[0]))
	print(np.max(res_arr))
	plt.figure()
	plt.title("corners-raw")
	plt.imshow(img_show, cmap=cm.gray)
	#opencv
	dst = cv2.cornerHarris(img_src,blockSize= 2,ksize= 3,k= 0.04) #blockSize为窗口大小，ksize为sobel算子的核大小，k为harris算子参数
	img_show2 = img_src.copy()
	if (len(img_show2.shape) == 2):
		img_show2 = cv2.cvtColor(img_show2, cv2.COLOR_GRAY2BGR)
	dst2 = dst.copy()
	dst2[dst<= 0.01* dst.max()]=0
	img_show2[dst2!=0] = (255, 0, 0)
	print(len(np.where(dst2 != 0)[0]))
	print(np.max(dst2))
	plt.figure()
	plt.title("opencv")
	plt.imshow(img_show2, cmap=cm.gray)
	#nms
	score_nms = corner_nms(res_arr)
	img_show3 = cv2.cvtColor(img_src,cv2.COLOR_GRAY2BGR)
	img_show3[score_nms != 0] = (255, 0, 0)
	plt.figure()
	plt.title("corners-nms")
	plt.imshow(img_show3, cmap=cm.gray)

	plt.show()


	print('end')





