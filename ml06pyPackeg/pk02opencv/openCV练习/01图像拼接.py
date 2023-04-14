import cv2
#从文件中读取图像
dir_root = r"D:\02dataset\01work\01TuoPanLJ\tuopan\0tmp_test/"
image1 = cv2.imread(dir_root + '1.bmp')
image2 = cv2.imread(dir_root + '2.bmp')
image3 = cv2.imread(dir_root + '3.bmp')
image4 = cv2.imread(dir_root + '4.bmp')
#将四个图像沿着水平方向拼接
horizontal_1 = cv2.hconcat([image1, image2])
cv2.imshow('Final Image horizontal_1', horizontal_1)
horizontal_2 = cv2.hconcat([image3, image4])
cv2.imshow('Final Image horizontal_1', horizontal_1)
#将两个拼接后的图像沿着垂直方向拼接
final_image = cv2.hconcat([horizontal_1, horizontal_2])
#显示拼接后的图像
cv2.imwrite('FinalImage.jpg', final_image)
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()