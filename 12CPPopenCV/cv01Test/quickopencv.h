#pragma once


#include <opencv2/opencv.hpp>

using namespace cv;

class QuickDemo {
public:
	void colorSpace_Demo(Mat& image);			//002.图像色彩空间转换
	void mat_creation_demo(/*Mat& image*/);		//003.图像对象的创建与赋值
	void pixel_visit_demo(Mat& image);			//004.图像像素的读写操作
	void operators_demo(Mat& image);			//005.图像像素的算术操作（加减乘除4种不同的API实现
	void tracking_bar_demo1(Mat& image);		//006.滚动条-调整图像亮度
	void tracking_bar_demo2(Mat& image);		//007.滚动条-传递参数
	void key_demo(Mat& image);					//008.键盘响应操作
	void color_style_demo(Mat& image);			//009.OpenCV自带颜色表操作
	void bitwise_demo(Mat& image);				//010.图像像素的逻辑操作（与，或，非，异或
	void channels_demo(Mat& image);				//011.通道合并与分离
	void inrange_demo(Mat& image);				//012.图像色彩空间转换（提取轮廓然后换绿幕
	void pixel_statistic_demo(Mat& image);		//013.图像像素值统计（min，max，mean均值，standard deviation标准方差
	void drawing_demo(Mat& image);				//014.图像几何形状绘制（圆，矩形，直线，椭圆
	void random_demo();							//015.随机数与随机颜色
	void polyline_drawing_demo(Mat& image);		//016.多边形填充与绘制
	void mouse_drawing_demo(Mat& image);		//017.鼠标操作与响应（提取选中的ROI区域
	void norm_demo(Mat& image);					//018.图像像素类型转换和归一化
	void resize_demo(Mat& image);				//019.图像放缩与插值
	void flip_demo(Mat& image);					//020.图像翻转
	void rotate_demo(Mat& image);				//021.图像旋转
	void video_demo1(Mat& image);				//022.视频文件摄像头使用
	void video_demo2(Mat& image);				//023.视频处理与保存
	void histogram_demo(Mat& image);			//024.图像直方图
	void histogram_2d_demo(Mat& image);			//025.二维直方图
	void histogram_eq_demo(Mat& image);			//026.直方图均衡化
	void blur_demo(Mat& image);					//027.图像卷积操作（会变模糊，且卷积核尺寸越大则越模糊
	void gaussian_blur_demo(Mat& image);		//028.高斯模糊
	void bifilter_demo(Mat& image);				//029.高斯双边模糊（可磨皮操作

	void Mask_operations_on_matrices();
	void Adding_two_images();
	void Changing_the_contrastand_brightness();

	void CV_001_imread();
	void CV_002_imwrite(Mat& pic);
	void CV_003_imshow(Mat& pic);
	void CV_005_shape(Mat& pic);

	
};