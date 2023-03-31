// 遍历图片
#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>

using namespace std;
using namespace cv;
cv::Mat inverseColor1(cv::Mat srcImage)
{
	cv::Mat tempImage = srcImage.clone();
	int row = tempImage.rows;
	int col = tempImage.cols;
	// 对各个像素点遍历进行取反操作
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			// 分别对各个通道进行反色处理
			tempImage.at<cv::Vec3b>(i, j)[0] = 255 - tempImage.at<cv::Vec3b>(i, j)[0];
			tempImage.at<cv::Vec3b>(i, j)[1] = 255 - tempImage.at<cv::Vec3b>(i, j)[1];
			tempImage.at<cv::Vec3b>(i, j)[2] = 255 - tempImage.at<cv::Vec3b>(i, j)[2];
		}
	}
	return tempImage;
}
cv::Mat inverseColor2(cv::Mat srcImage)
{
	cv::Mat tempImage = srcImage.clone();
	int row = tempImage.rows;
	// 由于图像是三通道图像，所以要作下面这样的处理
	int nStep = tempImage.cols * tempImage.channels();
	for (int i = 0; i < row; i++)
	{
		// 取源图像的第i行指针
		const uchar* pSrcData = srcImage.ptr<uchar>(i);
		// 取目标图像的第i行指针
		uchar* pResultData = tempImage.ptr<uchar>(i);
		for (int j = 0; j < nStep; j++)
		{
			pResultData[j] = cv::saturate_cast<uchar>(255 - pSrcData[j]);
		}
	}
	return tempImage;
}

cv::Mat inverseColor3(cv::Mat srcImage)
{
	int row = srcImage.rows;
	int col = srcImage.cols;
	cv::Mat tempImage = srcImage.clone();
	// 判断是否是连续图像，即是否有像素填充
	if (srcImage.isContinuous() && tempImage.isContinuous())
	{
		row = 1;
		// 按照行展开
		col = col * srcImage.rows * srcImage.channels();
	}
	// 遍历图像的每个像素
	for (int i = 0; i < row; i++)
	{
		// 设定图像数据源指针及输出图像数据指针
		const uchar* pSrcData = srcImage.ptr<uchar>(i);
		uchar* pResultData = tempImage.ptr<uchar>(i);
		for (int j = 0; j < col; j++)
		{
			*pResultData++ = 255 - *pSrcData++;
		}
	}
	return tempImage;
}

cv::Mat inverseColor4(cv::Mat srcImage)
{
	cv::Mat tempImage = srcImage.clone();
	// 初始化源图像迭代器
	cv::MatConstIterator_<cv::Vec3b> srcIterStart = srcImage.begin<cv::Vec3b>();
	cv::MatConstIterator_<cv::Vec3b> srcIterEnd = srcImage.end<cv::Vec3b>();
	// 初始化输出图像迭代器
	cv::MatIterator_<cv::Vec3b> resIterStart = tempImage.begin<cv::Vec3b>();
	cv::MatIterator_<cv::Vec3b> resIterEnd = tempImage.end<cv::Vec3b>();
	// 遍历图像反色处理
	while (srcIterStart != srcIterEnd)
	{
		(*resIterStart)[0] = 255 - (*srcIterStart)[0];
		(*resIterStart)[1] = 255 - (*srcIterStart)[1];
		(*resIterStart)[2] = 255 - (*srcIterStart)[2];
		// 迭代器递增
		srcIterStart++;
		resIterStart++;
	}
	return tempImage;
}
int test()
{
	cv::Mat img = cv::imread("testBig.jpg"); // 读入测试图像
	if (img.empty())
	{
		std::cout << "Cannot read image file" << std::endl;
		return -1;
	}
	int rows = img.rows;
	int cols = img.cols;
	int channels = img.channels();
	uchar* data = img.data; // 图像数据指针
	// 方法1 - 基于指针的方式
	std::clock_t start = std::clock();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			for (int k = 0; k < channels; k++)
			{
				uchar val = *(data + i * cols * channels + j * channels + k);
				// 读取像素值
			}
		}
	}
	std::printf("1:%.3f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	// 方法2 - 基于迭代器的方式
	start = std::clock();
	cv::MatIterator_<cv::Vec3b> it, end;
	for (it = img.begin<cv::Vec3b>(), end = img.end<cv::Vec3b>(); it != end; ++it)
	{
		cv::Vec3b val = *it;
		// 读取像素值
	}
	std::printf("2:%.3f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	// 方法3 - 基于at访问的方式
	start = std::clock();
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			cv::Vec3b val = img.at<cv::Vec3b>(i, j);
			// 读取像素值
		}
	}
	std::printf("3:%.3f s\n", (std::clock() - start) / (double)CLOCKS_PER_SEC);
	return 0;
}
int main()
{
	test();
	// 装载图像
	cv::Mat srcImage = cv::imread("test.jpg");
	cv::Mat dstImage;

	if (!srcImage.data)
		return -1;
	cv::imshow("srcImage", srcImage);

	dstImage = srcImage.clone();
	clock_t startTime, endTime;
	startTime = clock();//计时开始
	dstImage = inverseColor2(srcImage);
	endTime = clock();//计时结束
	cout << "The run time is: " << (endTime - startTime) << " ms" << endl;
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	startTime = clock();//计时开始
	dstImage = inverseColor1(srcImage);
	endTime = clock();//计时结束
	cout << "The run time is: " << (endTime - startTime) << " ms" << endl;
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	startTime = clock();//计时开始
	dstImage = inverseColor3(srcImage);
	endTime = clock();//计时结束
	cout << "The run time is: " << (endTime - startTime) << " ms" << endl;
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	startTime = clock();//计时开始
	dstImage = inverseColor4(srcImage);
	endTime = clock();//计时结束
	cout << "The run time is: " << (endTime - startTime) << " ms" << endl;
	cout << "The run time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	cv::imshow("dstImage", dstImage);

	cv::waitKey(0);
	return 0;
}