// 遍历图片
#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>
#include "CPostProcessor.h"
using namespace std;
using namespace cv;
void TraversalPicture();

int main()
{
	string dir_root = "D:/02dataset/01work/01TuoPanLJ/tuopan/black_0074690/";
	string img_first_name = "black_0074690_CM1_";
	vector<Mat> v_img;
	for (int i = 0; i < 4; i++) {
		string img_path = dir_root + img_first_name + to_string(i+1) + ".bmp";
		v_img.push_back(imread(img_path, cv::IMREAD_GRAYSCALE));
	}
	CPostProcessor pp = CPostProcessor();
	pp.Process(v_img);
	return 0;
}