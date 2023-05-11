#include <iostream>
#include<ctime>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "CReJudgeBack.h"
using namespace std;
using namespace cv;

cv::Mat getImgParaSample1() {
	cv::Mat img;
	string dir_root = "D:/02dataset/01work/05nanjingLG/06ReJudgeBack/testSimple/";
	string img_path = dir_root + "0.jpg";
	img = imread(img_path, cv::IMREAD_GRAYSCALE);
	return img;
}

void getImgDefectParaSample1(cv::Mat& v_img, vector<CDefect>& v_defect) {
	v_img = getImgParaSample1();
	CDefect defect = {{800,800},{900,900}, 6, 11, "ps"};
	v_defect.push_back(defect);
}

void RJBTestSimple() {
	cv::Mat img_gray;
	vector<CDefect> v_defect;
	getImgDefectParaSample1(img_gray, v_defect);

	CReJudgeBack pp = CReJudgeBack();
	bool result = pp.Process(img_gray, v_defect);
	cout << result << endl;
}