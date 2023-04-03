#include <iostream>
//#include <opencv2/opencv.hpp>
#include "quickopencv.h"


#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
using namespace std;
using namespace cv;

using namespace std;

void QuickDemo::colorSpace_Demo(Mat& image) {
	Mat gray, hsv; //定义2个矩阵类型的图像
	cvtColor(image, hsv, COLOR_BGR2HSV);	//转换成hdv （图像转换函数，第三个参数是转成的类型
	cvtColor(image, gray, COLOR_BGR2GRAY);	//转成灰度
	imshow("HSV", hsv);		//显示图片
	imshow("灰度", gray);	//显示图片
	//imwrite("F:\\文件夹\\C++\\OPENCV4入门学习\\图\\hsv.png", hsv);		//保存图片 （保存地址，保存图的名称）
	//imwrite("F:\\文件夹\\C++\\OPENCV4入门学习\\图\\gray.png", gray);	//保存
}

void QuickDemo::mat_creation_demo(/*Mat& image*/) {
	//Mat m1, m2;
	//m1 = image.clone();
	//image.copyTo(m2);

	//创建空白图形
	Mat m3 = Mat::ones(Size(400, 400), CV_8UC3);	//8位的无符号的3通道（改1则为单通道
	//ones 改 zeros则初始化为0
	//长度 = 通道数 * 宽度
	m3 = Scalar(255, 0, 0);		//给三个通道都赋值，单通道则 m3 = 127;
	//m3初始化为蓝色
	cout << "width:" << m3.cols << endl << "hight:" << m3.rows << endl << "channels:" << m3.channels() << endl;
	//显示宽度，长度，通道数
	//cout << m3 << endl;

	Mat m4;
	//m4 = m3;			//直接赋值 则m4变，m3也变（同体
	//m4 = m3.clone();	//m4为m3的克隆，m4变，m3不会变（不同体
	m3.copyTo(m4);		//把m3赋值给m4，m4为蓝色
	m4 = Scalar(0, 255, 255);	//改变m4的颜色为黄色
	imshow("图像3", m3);		//标题和图像名称  显示图像3 纯蓝色
	imshow("图像4", m4);
}

void QuickDemo::pixel_visit_demo(Mat& image) {
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;

	//数组下标访问像素值
	/*
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			if (dims == 1) {	//单通道的灰度图像
				int pv = image.at<uchar>(row, col);		//得到像素值
				image.at<uchar>(row, col) = 255 - pv;	//给像素值重新赋值（取反
			}
			if (dims == 3) {	//三通道的彩色图像
				Vec3b bgr = image.at<Vec3b>(row, col);	//opencv特定的类型，获取三维颜色，3个值
				image.at<Vec3b>(row, col)[0] = 255 - bgr[0];
				image.at<Vec3b>(row, col)[1] = 255 - bgr[1];
				image.at<Vec3b>(row, col)[2] = 255 - bgr[2]; //对彩色图像读取其像素值，并将其改写
			}
		}
	}
	*/

	//指针访问模式
	for (int row = 0; row < h; row++) {
		uchar* current_row = image.ptr<uchar>(row);
		for (int col = 0; col < w; col++) {
			if (dims == 1) {	//单通道的灰度图像
				int pv = image.at<uchar>(row, col);		//得到像素值
				*current_row++ = 255 - pv;	//给像素值重新赋值（取反
			}
			if (dims == 3) {	//三通道的彩色图像
				*current_row++ = 255 - *current_row;	//指针每做一次运算，就向后移动一位
				*current_row++ = 255 - *current_row;
				*current_row++ = 255 - *current_row;
			}

		}
	}
	namedWindow("像素读写演示", WINDOW_FREERATIO);
	imshow("像素读写演示", image);
	//imwrite("E:/2021.9.26备份/图片/Camera Roll/003颜色取反.png", image);	//保存
}

void QuickDemo::operators_demo(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	dst = image - Scalar(50, 50, 50);
	m = Scalar(50, 50, 50);

	multiply(image, m, dst);	//乘法操作 api
	imshow("乘法操作", dst);

	add(image, m, dst);			//加法操作 api
	imshow("加法操作", dst);

	subtract(image, m, dst);	//减法操作 api
	imshow("减法操作", dst);

	divide(image, m, dst);		//除法操作 api
	imshow("除法操作", dst);

	//加法操作底层
	/*
	int dims = image.channels();
	int h = image.rows;
	int w = image.cols;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w; col++) {
			Vec3b p1 = image.at<Vec3b>(row, col);	//opencv特定的类型，获取三维颜色，3个值
			Vec3b p2 = m.at<Vec3b>(row, col);
			dst.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(p1[0] + p2[0]);
			dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(p1[1] + p2[1]);
			dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(p1[2] + p2[2]);
		}
	}
	namedWindow("加法操作底层", WINDOW_FREERATIO);
	imshow("加法操作底层", dst);
	*/
}


Mat src, dst, m;
int lightness = 50;//定义初始化的亮度为50
static void on_track(int, void*) {
	m = Scalar(lightness, lightness, lightness);//创建调整亮度的数值
	add(src, m, dst);
	//subtract(src, m, dst);//定义亮度变换为减
	imshow("亮度调整", dst);//显示调整亮度之后的图片
}
void QuickDemo::tracking_bar_demo1(Mat &image) {
	namedWindow("亮度调整", WINDOW_AUTOSIZE);
	dst = Mat::zeros(image.size(), image.type());//图片的初始化创建一个和image大小相等，种类相同的图像
	m = Mat::zeros(image.size(), image.type());//图片的初始化创建一个和image大小相等，种类相同的图像
	src = image;//给src赋值
	int max_value = 100;//定义最大值为100
	createTrackbar("Value Bar", "亮度调整", &lightness, max_value, on_track);//调用函数实现功能
	on_track(50, 0);
}


static void on_lightness(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	m = Scalar(b, b, b);//创建调整亮度的数值
	addWeighted(image, 1.0, m, 0, b, dst);	//融合两张图 dst = image * 1.0 + m * 0 + b
	imshow(" 亮度&对比度调整  ", dst);//显示调整亮度之后的图片
}
static void on_contrast(int b, void* userdata) {
	Mat image = *((Mat*)userdata);
	Mat dst = Mat::zeros(image.size(), image.type());
	Mat m = Mat::zeros(image.size(), image.type());
	double contrast = b / 100.0;
	addWeighted(image, contrast, m, 0.0, 0, dst);
	imshow("  亮度&对比度调整  ", dst);//显示调整亮度之后的图片
}

void QuickDemo::tracking_bar_demo2(Mat& image) {
	namedWindow(" 亮度&对比度调整 ", WINDOW_AUTOSIZE);
	int lightness = 50;//定义初始化的亮度为50
	int max_value = 100;//定义最大值为100
	int contrast_value = 100;
	createTrackbar("Value Bar", " 亮度&对比度调整 ", &lightness, max_value, on_lightness, (void*)(&image));//调用函数实现功能
	//createTrackbar("Contrast Bar", " 亮度&对比度调整 ", &contrast_value, 200, on_contrast, (void*)(&image));//调用函数实现功能
	on_lightness(50, &image);
	//on_contrast(50,&image);
}

void QuickDemo::key_demo(Mat& image) {
	Mat dst = Mat::zeros(image.size(), image.type());
	while (true) {
		char c = waitKey(100);//等待100ms（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 退出应用程序
			break;
		}
		if (c == 49) {	//key#1
			cout << "you enter key #1" << endl;
			cvtColor(image, dst, COLOR_BGR2GRAY); //按键盘1，则转换后为灰度图像
		}
		if (c == 50) {	//key#2
			cout << "you enter key #2" << endl;
			cvtColor(image, dst, COLOR_BGR2HSV); //按键盘1，则转换后为HSV图像
		}
		if (c == 51) {	//key#3
			cout << "you enter key #3" << endl;
			dst = Scalar(50, 50, 50);
			cvtColor(image, dst, COLOR_BGR2HSV); //直接1到3会报错，则先转换为HSV图像
			add(image, dst, dst); //按键盘1，则转换后为增加亮度后的图像
		}
		imshow("键盘响应", dst);	//输出图像
	}
}

void QuickDemo::color_style_demo(Mat& image) {
	int colormap[] = {	//共19种
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_CIVIDIS,
		COLORMAP_DEEPGREEN,
		COLORMAP_HOT,
		COLORMAP_HSV,
		COLORMAP_INFERNO,
		COLORMAP_JET,
		COLORMAP_MAGMA,
		COLORMAP_OCEAN,
		COLORMAP_PINK,
		COLORMAP_PARULA,
		COLORMAP_RAINBOW,
		COLORMAP_SPRING,
		COLORMAP_TWILIGHT,
		COLORMAP_TURBO,
		COLORMAP_TWILIGHT,
		COLORMAP_VIRIDIS,
		COLORMAP_TWILIGHT_SHIFTED,
		COLORMAP_WINTER
	};
	Mat dst;
	int index = 0; //初始化为指向0的位置
	while (true) {
		char c = waitKey(500);//等待半秒（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 退出应用程序
			break;
		}
		if (c == 49) {	//key#1 按下按键1时。保存图片到指定位置
			cout << "you enter key #1" << endl;
			imwrite("E:/23_03_24_opencv_build/opencv450/butterfly-001.png", dst);
		}
		applyColorMap(image, dst, colormap[index % 19]);//循环展示19种图片（产生伪色彩图像）
		index++;
		cout << index<<endl;;
		imshow("循环播放", dst);
	}
}

void QuickDemo::bitwise_demo(Mat& image) {
	//绘制两张图
	Mat m1 = Mat::zeros(Size(256, 256), CV_8UC3);
	Mat m2 = Mat::zeros(Size(256, 256), CV_8UC3);
	rectangle(m1, Rect(100, 100, 80, 80), Scalar(255, 255, 0), -1, LINE_8, 0);//-1 =》小于0为填充，大于0为绘制
			   // Rect(左上角x，左上角y，矩形长，矩形宽)		    |=》搞锯齿的(表示四领域或者八领域的绘制
			   //最后的参数0表示中心坐标 或 半径坐标的小数位
	rectangle(m2, Rect(150, 150, 80, 80), Scalar(0, 255, 255), -1, LINE_8, 0);//小于0为填充，大于0为绘制
	imshow("m1", m1);
	imshow("m2", m2);
	//进行逻辑操作
	Mat dst;
	bitwise_and(m1, m2, dst);	//位操作 与
	imshow("像素位操作 与", dst);
	bitwise_or(m1, m2, dst);	//位操作 或
	imshow("像素位操作 或", dst);
	// dst = ~image;			//位操作 非（取反
	bitwise_not(image, dst);	//位操作 非（取反
	imshow("像素位操作 非", dst);
	bitwise_xor(m1, m2, dst);	//位操作 异或
	imshow("像素位操作 异或", dst);
}

void QuickDemo::channels_demo(Mat& image) {
	vector<Mat>mv;//可存放Mat类型的容器
	split(image, mv);//将多通道 拆分成 单通道（通道分离
	//imshow("蓝色", mv[0]);
	//imshow("绿色", mv[1]);
	//imshow("红色", mv[2]);

	// 三个通道分别为 B G R
	// 0，1，2 三个通道分别代表 B G R
	//关闭其中两个通道，则意味着 只开启剩余那个通道
	Mat dst;
	mv[0] = 0;
	mv[2] = 0;	// 关0，1则红色  关1，2则蓝色
	merge(mv, dst);//合并mv和dst
	imshow("绿色", dst);
	int from_to[] = { 0,2,1,1,2,0 };
	//把通道相互交换，第0->第2，第1->第1，第2->第0
	mixChannels(&image, 1, &dst, 1, from_to, 3);//3表示有3对要交换（即3个通道
	//参数为要进行混合的图像的地址，参数2为混合后图像的存放地址
	imshow("通道混合", dst);
	imshow("原图image不会变", image);
}
void QuickDemo::inrange_demo(Mat& image) {
	//提取任务的轮廓
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);//先把RGB色彩空间转换到hsv的空间中
	Mat mask;//其次提取图片的mask
	inRange(hsv, Scalar(35, 43, 46), Scalar(255, 255, 255), mask);//通过inRange提取hsv色彩空间的颜色
	//35，43，46根据图片表中的绿色最低来确定最小值（hmin,smin,vmim
	//77，255，255						    最大值
	//参数一地范围为，参数二高范围
	//将hsv中的由低到高的像素点提取出来并存储到mask中
	imshow("mask", mask);			//此时mask为白底
	Mat redback = Mat::zeros(image.size(), image.type());
	redback = Scalar(40, 40, 200);	//红色背景图
	bitwise_not(mask, mask);		//取反变成黑底
	imshow("mask", mask);
	image.copyTo(redback, mask);//将mask中不为0部分(白色像素点)对应的原图 拷贝到 redback上，mask通过inRange得到
	imshow("roi区域提取", redback);
}


void QuickDemo::pixel_statistic_demo(Mat& image) {
	double minv, maxv;
	Point minLoc, maxLoc;	//定义地址
	vector<Mat> mv;			//可存放Mat类型的容器
	split(image, mv);		//将多通道 拆分成 单通道（通道分离
	for (int i = 0; i < mv.size(); i++) {
		//分别打印各个通道的数值
		minMaxLoc(mv[i], &minv, &maxv, &minLoc, &maxLoc, Mat());//求出图像的最大值和最小值及其位置
		//参数一：输入单通道的数组		
		//参数二：返回最小值的指针			参数三：返回最大值的指针
		//参数四：返回最小值位置的指针		参数五：返回最大值位置的指针
		cout << "No.channels:" << i << "  minvalue:" << minv << "  maxvalue:" << maxv << endl;
	}
	Mat mean, stddev;
	meanStdDev(image, mean, stddev);//求出图像的均值的方差
	cout << "mean:" << mean << endl;
	cout << "stddev:" << stddev << endl;
}

void QuickDemo::drawing_demo(Mat& image) {
	Rect rect;				//矩形尺寸
	rect.x = 200;			//起始点x坐标
	rect.y = 200;			//起始点y坐标
	rect.width = 150;		//矩形宽度
	rect.height = 200;		//矩形高度
	Mat bg = Mat::zeros(image.size(), image.type());
	rectangle(bg, rect, Scalar(0, 0, 255), -1, 8, 0);								//画矩形
	//参数一：绘图的底图或画布名称   参数二：图片的起始，宽高
	//参数三：填充颜色				 参数四：>0为线宽，<0为填充
	//参数五：领域填充(控制边缘锯齿	 参数六：默认值为0
	circle(bg, Point(350, 400), 25, Scalar(0, 255, 0), 2, LINE_AA, 0);				//画圆
	//参数二：图片中心的位置		 参数三：表示圆的半径为25
	line(bg, Point(100, 100), Point(350, 400), Scalar(255, 0, 0), 8, LINE_AA, 0);	//画直线
	//参数二：线段起点坐标		 参数三：线段终点坐标		 LINE_AA表示去掉锯齿
	RotatedRect rrt;				//角度构造
	rrt.center = Point(200, 200);	//中心点位置
	rrt.size = Size(100, 200);		//x正沿x正方向，y正沿y正方向（可以是负的
	rrt.angle = 0.0;				//顺时针的角度（0-360度
	ellipse(bg, rrt, Scalar(255, 0, 255), 2, 8);									//画椭圆
	imshow("矩形，圆，直线，椭圆的绘制", bg);
}

void QuickDemo::random_demo() {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);//创建画布
	int w = canvas.cols;
	int h = canvas.rows;
	RNG rng(12345);		//产生随机数（12345为随机数的种子，默认的
	while (true) {
		char c = waitKey(10);//等待10ms（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 推出应用程序
			break;
		}
		int x1 = rng.uniform(0, canvas.cols);	//将随机坐标控制在画布范围内
		int y1 = rng.uniform(0, canvas.rows);
		int x2 = rng.uniform(0, w);
		int y2 = rng.uniform(0, h);
		int r = rng.uniform(0, 255);			//将随机颜色控制在255范围内
		int g = rng.uniform(0, 255);			//将随机颜色控制在255范围内
		int b = rng.uniform(0, 255);			//将随机颜色控制在255范围内
		//canvas = Scalar(0, 0, 0);				//想要每次都只出现一条线而不是叠加，则加上此句
		line(canvas, Point(x1, y1), Point(x2, y2), Scalar(r, g, b), 2, LINE_AA);	//画直线
		//参数二：线段起点坐标		参数三：线段终点坐标	2为线宽		 LINE_AA表示去掉锯齿
		imshow("随机绘制演示", canvas);
	}
}

void QuickDemo::polyline_drawing_demo(Mat& image) {
	Mat canvas = Mat::zeros(Size(512, 512), CV_8UC3);
	Point p1(150, 100);		//第一个点的坐标
	Point p2(350, 200);		//  二
	Point p3(240, 300);		//  三
	Point p4(150, 300);		//  四
	Point p5(50, 200);		//  五
	vector<Point> pts;							//搞一个容器，用来装 点
	pts.push_back(p1);		//将点放进容器内
	pts.push_back(p2);		//因 未初始化数组容量，所以要用 push_back 操作
	pts.push_back(p3);		//若 已初始化，可以用 数组下标 来操作
	pts.push_back(p4);
	pts.push_back(p5);
	//fillPoly(canvas, pts, Scalar(122, 155, 255), 8, 0);				//填充多边形
	// polylines(canvas, pts, true, Scalar(90, 0, 255), 5, 8, 0);		//绘制多边形
	//参数一：画布				参数二：点集			参数三：一定要写true（封闭图形
	//参数倒3：线宽(最少为1		参数倒2：线的渲染方式	参数倒1：相对左上角（0,0）的位置

	//单个API搞定多边形的绘制和填充
	vector<vector<Point>> contours;				//搞一个容器，用来装 多边形的点集
	contours.push_back(pts);		//将一个多边形的点集放进容器内，作为一个元素
	drawContours(canvas, contours, -1, Scalar(0, 0, 255), -1);			//参数倒1：<0表示填充，>0表示线宽
	//参数二：多边形的点集		参数三：-1为绘制全部的多边形；0为绘制第一个，1为绘制第二个，以此类推
	imshow("多边形绘制", canvas);
}

//选中的矩形区域提取
Point sp(-1, -1);	//鼠标的起始位置
Point ep(-1, -1);	//鼠标的结束位置
Mat temp;
static void on_draw(int event, int x, int y, int flags, void* userdata) {
	//参数一（event）为鼠标事件
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN) {			//若鼠标的左键按下
		sp.x = x;
		sp.y = y;		//此时鼠标的起始位置坐标
		cout << "start point" << sp << endl;
	}
	else if (event == EVENT_LBUTTONUP) {		//若鼠标的左键抬起
		ep.x = x;
		ep.y = y;		//此时鼠标的结束位置坐标
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {		//若鼠标有移动过
			Rect box(sp.x, sp.y, dx, dy);
			imshow("ROI区域", image(box));
			rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			imshow("鼠标绘制", image);				//这里是为了显示结果
			sp.x = -1;	//复位，为下一次做准备
			sp.y = -1;	//复位，为下一次做准备
		}
	}
	else if (event == EVENT_MOUSEMOVE) {		//若鼠标正在移动
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;		//此时鼠标的结束位置坐标
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {		//若鼠标有移动过
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);	//为了不将鼠标移动过程中的框也显示出来
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);			//这里是为了每次重新提取都将前面的覆盖
			}
		}
	}
}
void QuickDemo::mouse_drawing_demo(Mat& image) {
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	//设置窗口是回调函数，参数二表示调用on_draw
	imshow("鼠标绘制", image);
	temp = image.clone();
}

void QuickDemo::norm_demo(Mat& image) {
	Mat dst;
	cout << image.type() << endl;				//打印图片的类型
	image.convertTo(image, CV_32F);				//将image的数据转换成浮点型float32位数据
	cout << image.type() << endl;				//打印转换后的图片数据类型
	normalize(image, dst, 1.0, 0, NORM_INF);	//进行归一化操作
	//参数一：要进行归一化的图片	参数二：归一化后要输出的图片
	//参数三：alpha					参数四：beta			参数五：归一化方法
	cout << dst.type() << endl;					//打印归一化后的图像的类型
	imshow("图像的归一化", dst);				//显示归一化后的图像
	//CV_8UC3   原本为 3通道，每个通道8位的UC（无符号）类型
	//CV_32FC3  转换后 3通道，每个通道32位的浮点数类型
	/*
	归一化方法：
	NORM_L1（依据sum）				b不用，a为归一化后矩阵的范数值
	NORM_L2（依据单位向量为1）		b不用，a为 同上
	NORM_MINMAX（依据最大值）		b不用，a为 同上
	NORM_INF（依据min与max的差值）	a为归一化后的最小值，b归一化后的最大值
	*/
}

void QuickDemo::resize_demo(Mat& image) {
	Mat zoomin, zoomout;
	int h = image.rows;
	int w = image.cols;
	resize(image, zoomout, Size(w / 2, h / 2), 0, 0, INTER_LINEAR);		// INTER_LINEAR 为线性插值
	//若Size里的值没变，则按照参数四fx（水平轴）和参数五fy（垂直轴）来进行放缩操作
	//参数六：插值的方法
	imshow("zoomout", zoomout);
	resize(image, zoomin, Size(w * 1.5, h * 1.5), 0, 0, INTER_LINEAR);
	imshow("zoomin", zoomin);

	//double time = (1920 / w) > (1080 / h) ? (1080 / h) : (1920 / w);
	//double b = 1.0;
	//while (true) {
	//	char c = waitKey(100);//等待100ms（1s = 1000ms），做视频处理都是1
	//	if (c == 27) {	//按 esc 退出应用程序
	//		break;
	//	}
	//	if (c == 49) {	//key#1
	//		cout << "you enter key #1" << endl;
	//		b=(b + 0.1)>time?time:(b+0.1);
	//		//cvtColor(image, dst, COLOR_BGR2GRAY); //按键盘1，则转换后为灰度图像
	//		resize(image, zoomout, Size(w * b, h * b), 0, 0, INTER_LINEAR);
	//	}
	//	if (c == 50) {	//key#2
	//		cout << "you enter key #2" << endl;
	//		b = (b - 0.1) < 0.5 ? 0.5 : (b - 0.1);
	//		//cvtColor(image, dst, COLOR_BGR2HSV); //按键盘1，则转换后为HSV图像
	//		resize(image, zoomout, Size(w * b, h * b), 0, 0, INTER_LINEAR);
	//	}
	//	imshow("键盘响应", zoomout);	//输出图像
	//}
}

void QuickDemo::flip_demo(Mat& image) {
	Mat dst;
	flip(image, dst, 0);			// 0 上下翻转 x对称
	imshow("图像上下翻转", dst);
	flip(image, dst, 1);			// 1 左右翻转 y对称
	imshow("图像左右翻转", dst);
	flip(image, dst, -1);			//-1 上下左右都翻转（相当于旋转180°）
	imshow("图像上下左右翻转", dst);
}

void QuickDemo::rotate_demo(Mat& image) {
	Mat dst, M;				//M为2*3的变换矩阵（旋转矩阵）
	int w = image.cols;		//图片宽度
	int h = image.rows;		//图片高度
	M = getRotationMatrix2D(Point(w / 2, h / 2), 45, 1.0);	//获得旋转矩阵 M
	//参数一：原来图像的中心点位置		参数二：旋转角度(逆时针)	参数三：图像本身大小的放大缩小
	double cos = abs(M.at<double>(0, 0));	//取绝对值
	double sin = abs(M.at<double>(0, 1));
	/*
		[x'] = [ cos  sin] * [x]
		[y']   [-sin  cos]   [y],
			M =	[ cos  sin  0]
				[-sin  cos  0], （第三列用来控制平移）
	*/
	double nw = cos * w + sin * h;		//旋转后图像所占矩形的宽
	double nh = sin * w + cos * h;		//旋转后图像所占矩形的高
	//更新 新的中心  （将新中心平移到正确位置上）
	M.at<double>(0, 2) += (nw / 2 - w / 2);		//将矩形的宽高 加上偏差量  （新M的第一列最后的值）
	M.at<double>(1, 2) += (nh / 2 - h / 2);		//将矩形的宽高 加上偏差量  （新M的第二列最后的值）
	warpAffine(image, dst, M, Size(nw, nh), INTER_LINEAR, 0, Scalar(255, 255, 0));	//进行旋转
	//参数四：原来图像的中心点位置		参数五：插值方式
	//参数六：边缘的处理方式			参数七：边缘底图的颜色
	//namedWindow("旋转演示", WINDOW_FREERATIO); //可调整显示图片的窗口大小
	imshow("旋转演示", dst);
}

void QuickDemo::video_demo1(Mat& image) {
	//读已有视频
	VideoCapture capture("E:/2021.9.26备份/图片/Camera Roll/人脸素材.mp4");//读取视频地址
	Mat frame;	//定义一个二值化的 frame
	while (true) {
		capture.read(frame);
		//flip(frame, frame, 1);			// 1 左右翻转 y对称 （镜像）
		if (frame.empty())	//如果读入失败
		{
			break;	//若视频为空，则跳出操作
		}
		imshow("frame", frame);			//显示视频
		colorSpace_Demo(frame);			//对视频调用之前的demo
		int c = waitKey(1);			//等待10ms（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 退出应用程序
			break;
		}
	}
	capture.release();	//释放相机的资源
	/*
	//调用电脑摄像头
	VideoCapture capture(0);
	Mat frame;	//定义一个二值化的 frame
	while (true) {
		capture.read(frame);
		if (frame.empty())	//如果读入失败
		{
			break;	//若视频为空，则跳出操作
		}
		flip(frame, frame, 1);			// 1 左右翻转 y对称 （镜像）
		imshow("frame", frame);			//显示视频
		int c = waitKey(10);			//等待10ms（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 退出应用程序
			break;
		}
	}
	*/
}

void QuickDemo::video_demo2(Mat& image) {
	//视频的属性：SD(标清)，HD(高清)，UHD(超清)，蓝光。
	VideoCapture capture("E:/2021.9.26备份/图片/Camera Roll/人脸素材.mp4");//读取视频地址
	int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);	//获取视频的宽度
	int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);	//获取视频的高度
	int count = capture.get(CAP_PROP_FRAME_COUNT);			//获取视频总的帧数
	//fps是衡量处理视频的能力 （一秒钟处理多少张图片的能力，处理速度越快则越好）
	double fps = capture.get(CAP_PROP_FPS);
	cout << "frame width：" << frame_width << endl;
	cout << "frame height：" << frame_height << endl;
	cout << "FPS：" << fps << endl;
	cout << "Number of frame：" << count << endl;
	VideoWriter writer("F:/文件夹/C++/OPENCV4入门学习/图/test.mp4", capture.get(CAP_PROP_FOURCC), fps, Size(frame_width, frame_height), true);
	//参数一：保存地址		参数二：获取图片的格式(编码方式)		参数三：图片是帧数		参数四：视频宽高		参数五：与原来颜色保持一致
	//等全部运行完再去查看视频是否保存成功
	Mat frame;
	while (true) {
		capture.read(frame);
		//flip(frame, frame, 1);			// 1 左右翻转 y对称 （镜像）
		if (frame.empty())	//如果读入失败
		{
			break;	//若视频为空，则跳出操作
		}
		imshow("frame", frame);			//显示视频
		colorSpace_Demo(frame);			//对视频调用之前的demo
		writer.write(frame);
		int c = waitKey(1);			//等待10ms（1s = 1000ms），做视频处理都是1
		if (c == 27) {	//按 esc 退出应用程序
			break;
		}
	}
	//release
	writer.release();
	capture.release();	//释放相机的资源
}

void QuickDemo::histogram_demo(Mat& image) {
	//三通道分离
	vector<Mat> bgr_plane;
	split(image, bgr_plane);
	//定义参数变量
	const int channels[1] = { 0 };
	const int bins[1] = { 256 };	//总共 256 个灰度级别
	float hranges[2] = { 0,255 };	//每个通道的取值范围是 0 到 255
	const float* ranges[1] = { hranges };
	Mat b_hist;
	Mat g_hist;
	Mat r_hist;
	//计算 Blue，Green，Red 通道的直方图
	calcHist(&bgr_plane[0], 1, 0, Mat(), b_hist, 1, bins, ranges);	//第一个通道
	calcHist(&bgr_plane[1], 1, 0, Mat(), g_hist, 1, bins, ranges);
	calcHist(&bgr_plane[2], 1, 0, Mat(), r_hist, 1, bins, ranges);
	//参数一：要计算直方图的数据					参数二：1表示只有一张图(输入图像的格式)
	//参数三：需要统计直方图的第几个通道			参数四：掩模，mask必须是8位的数组且和参数一的大小一致
	//参数五：b_hist表示直方图的输出				参数六：1表示维度是一维的(输出直方图的维度dims)
	//参数七：直方图中每个维度需分成的区间个数		参数八：ranges表示直方图的取值范围(区间)

//显示直方图
	int hist_w = 512;										//设置 画布宽度 为512
	int hist_h = 400;										//设置 画布高度 为400
	int bin_w = cvRound((double)hist_w / bins[0]);			//每个 bin 占的宽度
			  //cvRound()四舍五入返回数值
	Mat histImage = Mat::zeros(hist_h, hist_w, CV_8UC3);	//创建画布

	//归一化直方图数据（归一化到大小一致的范围内）
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());	//histImage.rows是为了不超出画布许可的高度范围
	normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	//参数一：要进行归一化的图片	参数二：归一化后要输出的图片
	//参数三：alpha					参数四：beta			参数五：归一化方法

//绘制直方图曲线
	for (int i = 1; i < bins[0]; i++) {		//每个bin占2个像素的位置
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 3, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 3, 0);
		line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 3, 0);
		//从前一个位置到当前位置连上一条线
	}
	//显示直方图
	namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	imshow("Histogram Demo", histImage);
}

void QuickDemo::histogram_2d_demo(Mat& image) {
	//2D直方图
	Mat hsv, hs_hist;
	cvtColor(image, hsv, COLOR_BGR2HSV);	//先把RGB色彩空间转换到hsv的空间中
	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins, sbins };		//h和s这两个维度需分成的 区间个数
	float h_range[] = { 0,180 };			//h的取值范围
	float s_range[] = { 0,256 };			//s的取值范围
	const float* hs_ranges[] = { h_range, s_range };
	int hs_channels[] = { 0,1 };
	//计算通道的直方图
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
	//参数一：要计算直方图的数据					参数二：1表示只有一张图(输入图像的格式)
	//参数三：需要统计直方图的第几个通道(前两个)	参数四：掩模，mask必须是8位的数组且和参数一的大小一致
	//参数五：b_hist表示直方图的输出				参数六：2表示维度是二维的(输出直方图的维度dims)
	//参数七：直方图中每个维度需分成的区间个数		参数八：hs_ranges表示直方图的取值范围(区间)
	//参数九：是否对得到的直方图进行归一化处理		参数十：在多个图像时，是否累计计算像素值的个数
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);	//寻找最大值和最小值及其位置（这里先找到最大值）
	//参数一：输入单通道的数组		
	//参数二：返回最小值的指针			参数三：返回最大值的指针
	//参数四：返回最小值位置的指针		参数五：返回最大值位置的指针
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins * scale, hbins * scale, CV_8UC3);	//创建空白图像
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++) {
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h * scale, s * scale),
				Point((h + 1) * scale - 1, (s + 1) * scale - 1), Scalar::all(intensity), -1);
		}
	}
	//显示直方图
	//applyColorMap(hist2d_image, hist2d_image, COLORMAP_JET);	//产生伪色彩图像
	namedWindow("H-S Histogram", WINDOW_AUTOSIZE);
	imshow("H-S Histogram", hist2d_image);
	//imwrite("F:/文件夹/C++/OPENCV4入门学习/图/hist_2d.png", hist2d_image);
}

void QuickDemo::histogram_eq_demo(Mat& image) {
	//直方图均衡化 (目的是对比度拉伸，即 对比度会更强)
	//用途：用于图像增强，人脸检测，卫星遥感(提升图像质量)。
	//opencv中，均衡化的图像只支持单通道
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	imshow("灰度图像", gray);
	Mat dst;
	equalizeHist(gray, dst);
	imshow("直方图均衡化演示", dst);
}

void QuickDemo::blur_demo(Mat& image) {	//会变模糊，且卷积核尺寸越大则越模糊
	Mat dst;
	blur(image, dst, Size(15, 15), Point(-1, -1));	//均值滤波 均值模糊
	//参数三：卷积核的大小		参数四：卷积的起始点（Point(-1, -1)则默认取核的中心）
	//	参数三中：
	//Size(15, 1) 左右晃动的模糊（只有行的话
	//Size(1, 15) 上下			（	  列
	imshow("图像卷积操作", dst);
}

void QuickDemo::gaussian_blur_demo(Mat& image) {
	//中心值最大，离中心越远值越小
	Mat dst;
	GaussianBlur(image, dst, Size(5, 5), 15);
	//参数三：高斯矩阵的大小（正数且奇数）
	//参数四：sigmaX 和 sigmaY 为15 
	//（参数三和四都 值越大则越模糊，且参数四的影响更明显）
	imshow("高斯模糊", dst);
}

void QuickDemo::bifilter_demo(Mat& image) { //可做磨皮操作
	Mat dst;
	bilateralFilter(image, dst, 0, 100, 10);
	//参数三：色彩空间		参数四：坐标空间	（双边是指 色彩空间 和 坐标空间
	namedWindow("高斯双边模糊", WINDOW_AUTOSIZE);
	imshow("高斯双边模糊", dst);
}

static void help(char* progName)
{
	cout << endl
		<< "This program shows how to filter images with mask: the write it yourself and the"
		<< "filter2d way. " << endl
		<< "Usage:" << endl
		<< progName << " [image_path -- default lena.jpg] [G -- grayscale] " << endl << endl;
}
void Sharpen(const Mat& myImage, Mat& Result)
{
	CV_Assert(myImage.depth() == CV_8U);  // accept only uchar images
	const int nChannels = myImage.channels();
	Result.create(myImage.size(), myImage.type());
	for (int j = 1; j < myImage.rows - 1; ++j)
	{
		const uchar* previous = myImage.ptr<uchar>(j - 1);
		const uchar* current = myImage.ptr<uchar>(j);
		const uchar* next = myImage.ptr<uchar>(j + 1);
		uchar* output = Result.ptr<uchar>(j);
		for (int i = nChannels; i < nChannels * (myImage.cols - 1); ++i)
		{
			*output++ = saturate_cast<uchar>(5 * current[i]
				- current[i - nChannels] - current[i + nChannels] - previous[i] - next[i]);
		}
	}
	Result.row(0).setTo(Scalar(0));
	Result.row(Result.rows - 1).setTo(Scalar(0));
	Result.col(0).setTo(Scalar(0));
	Result.col(Result.cols - 1).setTo(Scalar(0));
}

void QuickDemo::Mask_operations_on_matrices()
{
	const char* filename = "E:/23_03_24_opencv_build/opencv450/butterfly-3.png";
	Mat src, dst0, dst1;
	//if (argc >= 3 && !strcmp("G", argv[2]))
	//	src = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
	//else
		src = imread(samples::findFile(filename), IMREAD_COLOR);
	if (src.empty())
	{
		cerr << "Can't open image [" << filename << "]" << endl;
		return ;
	}
	namedWindow("Input", WINDOW_AUTOSIZE);
	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Input", src);
	double t = (double)getTickCount();
	Sharpen(src, dst0);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Hand written function time passed in seconds: " << t << endl;
	imshow("Output", dst0);
	waitKey();
	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0,
		-1, 5, -1,
		0, -1, 0);
	t = (double)getTickCount();
	filter2D(src, dst1, src.depth(), kernel);
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "Built-in filter2D time passed in seconds:     " << t << endl;
	imshow("Output", dst1);
}

void QuickDemo::Adding_two_images()
{
	double alpha = 0.5; double beta; double input;
	Mat src1, src2, dst;
	//cout << " Simple Linear Blender " << endl;
	//cout << "-----------------------" << endl;
	//cout << "* Enter alpha [0.0-1.0]: ";
	//cin >> input;
	//// We use the alpha provided by the user if it is between 0 and 1
	//if (input >= 0 && input <= 1)
	//{
	//	alpha = input;
	//}
	src1 = imread(("E:/23_03_24_opencv_build/opencv-470-src/4.7.0/opencv-4.7.0/samples/data/LinuxLogo.jpg"));
	src2 = imread(("E:/23_03_24_opencv_build/opencv-470-src/4.7.0/opencv-4.7.0/samples/data/WindowsLogo.jpg"));
	if (src1.empty()) { cout << "Error loading src1" << endl; return ; }
	if (src2.empty()) { cout << "Error loading src2" << endl; return ; }
	beta = (1.0 - alpha);
	addWeighted(src1, alpha, src2, beta, 0.0, dst);
	imshow("Linear Blend", dst);
	waitKey(0);
	return;
}

void QuickDemo::Changing_the_contrastand_brightness()
{
	Mat image = imread(samples::findFile(("E:/23_03_24_opencv_build/opencv-470-src/4.7.0/opencv-4.7.0/samples/data/lena.jpg")));
	if (image.empty())
	{
		cout << "Could not open or find the image!\n" << endl;
		return ;
	}
	Mat new_image = Mat::zeros(image.size(), image.type());
	double alpha = 1.0; /*< Simple contrast control */
	int beta = 0;       /*< Simple brightness control */
	cout << " Basic Linear Transforms " << endl;
	cout << "-------------------------" << endl;
	cout << "* Enter the alpha value [1.0-3.0]: "; cin >> alpha;
	cout << "* Enter the beta value [0-100]: ";    cin >> beta;
	//for (int y = 0; y < image.rows; y++) {
	//	for (int x = 0; x < image.cols; x++) {
	//		for (int c = 0; c < image.channels(); c++) {
	//			new_image.at<Vec3b>(y, x)[c] =
	//				saturate_cast<uchar>(alpha * image.at<Vec3b>(y, x)[c] + beta);
	//		}
	//	}
	//}
	image.convertTo(new_image, -1, alpha, beta);

	imshow("Original Image", image);
	imshow("New Image", new_image);
	waitKey();
	return;
}

void QuickDemo::CV_001_imread()
{
	/*
	函数说明：

retval = cv.imread(filename[, flags])

函数 cv2.imread() 从指定文件加载图像并返回该图像的矩阵。
如果无法读取图像（文件丢失，权限不正确，格式不支持或无效），该函数返回一个空矩阵。
目前支持的文件格式：
Windows 位图 - * .bmp，* .dib
JPEG 文件 - * .jpeg，* .jpg，*.jpe
JPEG 2000文件 - * .jp2
便携式网络图形 - * .png
WebP - * .webp
便携式图像格式 - * .pbm，* .pgm，* .ppm * .pxm，* .pnm
TIFF 文件 - * .tiff，* .tif
参数说明：

filename：读取图像的文件路径和文件名
flags：读取图片的方式，可选项
cv2.IMREAD_COLOR(1)：始终将图像转换为 3 通道BGR彩色图像，默认方式
cv2.IMREAD_GRAYSCALE(0)：始终将图像转换为单通道灰度图像
cv2.IMREAD_UNCHANGED(-1)：按原样返回加载的图像（使用Alpha通道）
cv2.IMREAD_ANYDEPTH(2)：在输入具有相应深度时返回16位/ 32位图像，否则将其转换为8位
cv2.IMREAD_ANYCOLOR(4)：以任何可能的颜色格式读取图像
返回值 retval：读取的 OpenCV 图像，nparray 多维数组
注意事项：

OpenCV 读取图像文件，返回值是一个nparray 多维数组。OpenCV 对图像的任何操作，本质上就是对 Numpy 多维数组的运算。
OpenCV 中彩色图像使用 BGR 格式，而 PIL、PyQt、matplotlib 等库使用的是 RGB 格式。
cv2.imread() 如果无法从指定文件读取图像，并不会报错，而是数返回一个空矩阵。
cv2.imread() 指定图片的存储路径和文件名，在 python3 中不支持中文和空格（但并不会报错）。必须使用中文时，可以使用 cv2.imdecode() 处理，参见扩展例程。
cv2.imread() 读取图像时默认忽略透明通道，但可以使用 CV_LOAD_IMAGE_UNCHANGED 参数读取透明通道。
对于彩色图像，可以使用 flags=0 按照读取为灰度图像。
	*/
	const char* imagename = "E:/23_03_24_opencv_build/opencv450/butterfly.png";
	Mat pic = imread(imagename, IMREAD_COLOR);
}

void QuickDemo::CV_002_imwrite(Mat& pic)
{
	/*
	函数 cv2.imwrite() 用于将图像保存到指定的文件。

函数说明：

retval = cv2.imwrite(filename, img [, paras])

cv2.imwrite() 将 OpenCV 图像保存到指定的文件。
cv2.imwrite() 基于保存文件的扩展名选择保存图像的格式。
cv2.imwrite() 只能保存 BGR 3通道图像，或 8 位单通道图像、或 PNG/JPEG/TIFF 16位无符号单通道图像。
参数说明：

filename：要保存的文件的路径和名称，包括文件扩展名

img：要保存的 OpenCV 图像，nparray 多维数组

paras：不同编码格式的参数，可选项

cv2.CV_IMWRITE_JPEG_QUALITY：设置 .jpeg/.jpg 格式的图片质量，取值为 0-100（默认值 95），数值越大则图片质量越高；
cv2.CV_IMWRITE_WEBP_QUALITY：设置 .webp 格式的图片质量，取值为 0-100；
cv2.CV_IMWRITE_PNG_COMPRESSION：设置 .png 格式图片的压缩比，取值为 0-9（默认值 3），数值越大则压缩比越大。
retval：返回值，保存成功返回 True，否则返回 False。

注意事项：

cv2.imwrite() 保存的是 OpenCV 图像（多维数组），不是 cv2.imread() 读取的图像文件，所保存的文件格式是由 filename 的扩展名决定的，与读取的图像文件的格式无关。
对 4 通道 BGRA 图像，可以使用 Alpha 通道保存为 PNG 图像。
cv2.imwrite() 指定图片的存储路径和文件名，在 python3 中不支持中文和空格（但并不会报错）。必须使用中文时，可以使用 cv2.imencode() 处理，参见扩展例程。
————————————————
版权声明：本文为CSDN博主「youcans_」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/youcans/article/details/121169014
	*/
	if (pic.empty()) {
		return;
	}

	const char* imagename = "E:/23_03_24_opencv_build/opencv450/butterfly-3.png";
	bool b = imwrite(imagename, pic);
}

void QuickDemo::CV_003_imshow(Mat& pic)
{
	if (pic.empty()) {
		return;
	}
	imshow("pic", pic);

	namedWindow("pic-win", WINDOW_NORMAL);
	imshow("pic-win",pic);
}

void QuickDemo::CV_005_shape(Mat& pic)
{
	if (pic.empty()) {
		return;
	}
	int dims=pic.dims;
}
