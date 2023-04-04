// 遍历图片
#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>
#include "CPostProcessor.h"
#include "quickopencv.h"
//#include <fstream>
//#include <nlohmann/json.hpp>
//using json = nlohmann::json;
using namespace std;
using namespace cv;
void TraversalPicture();
void cvtest()
{
	std::cout << "Hello World!\n";
	string imagename = "E:/23_03_24_opencv_build/opencv450/butterfly.png"; 
	Mat img = imread(imagename, IMREAD_COLOR);	//彩色
	//Mat img = imread(imagename, IMREAD_GRAYSCALE);	//黑白

	//如果读入图像失败
	if (img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename);
	}
	else {
		namedWindow("image", WINDOW_FREERATIO);
		imshow("image", img);
	}
	QuickDemo pd;//创建类对象
	pd.polyline_drawing_demo(img);

	waitKey();
	destroyAllWindows();
}
/*
void jsonTest() {
	string json_name = "F:/sheepy/result.json";
	std::ifstream f(json_name);
	json data = json::parse(f);
	auto points = data["regions"][0]["polygon"]["outer"]["point"];
	for (int i = 0; i < 4; ++i) {
		auto point = points[i];
		float x = point["x"];
		Point p = { point["x"] , point["y"] };
		cout << point["x"] << point["y"] << endl;
	}
}
*/
void PPtest()
{
	//string dir_root = "F:/sheepy/02data/01LG/test/";
	string dir_root = "D:/02dataset/01work/01TuoPanLJ/tuopan/black_0074690/";
	string img_first_name = "black_0074690_CM1_";
	vector<Mat> v_img;
	for (int i = 0; i < 4; i++) {
		string img_path = dir_root + img_first_name + to_string(i + 1) + ".bmp";
		//v_img.push_back(imread(img_path, cv::IMREAD_GRAYSCALE));
		v_img.push_back(imread(img_path));
	}
	CPostProcessor pp = CPostProcessor();
	vector<vector<CDefect>> vv_defect;
	vector<CDefect> v_defect = { {{450,700},{700,1630}, 6, 1 } };
	vv_defect.push_back(v_defect);
	vv_defect.push_back(v_defect);
	vv_defect.push_back(v_defect);
	vv_defect.push_back(v_defect);
	pp.Process(v_img, vv_defect);
}
int main()
{
	PPtest();
	return 0;
}