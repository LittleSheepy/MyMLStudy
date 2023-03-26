//出处：昊虹AI笔记网(hhai.cc)
//用心记录计算机视觉和AI技术

//博主微信/QQ 2487872782
//QQ群 271891601
//欢迎技术交流与咨询

//OpenCV版本 OpenCV3.0

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

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

int main()
{
	// 装载图像
	cv::Mat srcImage = cv::imread("F:/material/images/P0028-flower-02.jpg");
	cv::Mat dstImage;

	if (!srcImage.data)
		return -1;
	cv::imshow("srcImage", srcImage);

	dstImage = srcImage.clone();

	dstImage = inverseColor3(srcImage);

	cv::imshow("dstImage", dstImage);

	cv::waitKey(0);
	return 0;
}