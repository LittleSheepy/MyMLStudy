import cv2

img_path = r"D:\00work\0code\bin\img_save\NumRecDebug/202304221547_img_num_little_4.bmp"
template0_path = r"D:\02dataset\01work\05nanjingLG\03NumRec\template/0.bmp"
template9_path = r"D:\02dataset\01work\05nanjingLG\03NumRec\template/9.bmp"
img = cv2.imread(img_path)
template0 = cv2.imread(template0_path)
template9 = cv2.imread(template9_path)
print("")
print("TM_CCOEFF_NORMED")
res = cv2.matchTemplate(img, template0, cv2.TM_CCOEFF_NORMED)
max_ = res.max()
print(max_)

res = cv2.matchTemplate(img, template9, cv2.TM_CCOEFF_NORMED)
max_ = res.max()
print(max_)

print("TM_CCORR_NORMED")
res = cv2.matchTemplate(img, template0, cv2.TM_CCORR_NORMED)
min_ = res.min()
print(min_)
max_ = res.max()
print(max_)

res = cv2.matchTemplate(img, template9, cv2.TM_CCORR_NORMED)
min_ = res.min()
print(min_)
max_ = res.max()
print(max_)

print("TM_SQDIFF_NORMED")
res = cv2.matchTemplate(img, template0, cv2.TM_SQDIFF_NORMED)
min_ = res.min()
print(min_)

res = cv2.matchTemplate(img, template9, cv2.TM_SQDIFF_NORMED)
min_ = res.min()
print(min_)