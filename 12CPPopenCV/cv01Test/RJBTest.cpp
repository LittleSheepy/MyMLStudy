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
	//string dir_root = "D:/02dataset/01work/05nanjingLG/06ReJudgeBack/testSimple/";
	string dir_root = "./PostProcessTest/ReJudgeBackTest/";
	string img_path = dir_root + "BackRegion.jpg";
	img = imread(img_path, cv::IMREAD_GRAYSCALE);
	return img;
}

void getImgDefectParaSample1(cv::Mat& v_img, vector<CDefect>& v_defect) {
	v_img = getImgParaSample1();
	CDefect defect = { {800,40},{260,50}, 100, 10, "LBPS", 1.99 };
	v_defect.push_back(defect);
}

void getImgDefectParaBySerial(cv::Mat& img, vector<CDefect>& v_defect, int serial = 1) {
	const std::string folder_path = "./PostProcessTest/ReJudgeBackTest/";
	// 读取图片
	/*Mat img = v_img[i];*/
	std::string filename = folder_path + std::to_string(serial);
	//std::string img_name = filename + ".bmp";
	//img = imread(img_name);

	// 读取 vector<CDefect>
	std::string vect_name = filename + ".txt";
	std::ifstream outputFile(vect_name, std::ios::binary);
	CDefect defect;
	while (!outputFile.eof()) {
		outputFile >> defect.p1.x;
		if (outputFile.eof()) {
			break;
		}
		outputFile >> defect.p1.y;
		outputFile >> defect.p2.x >> defect.p2.y;
		outputFile >> defect.area;
		outputFile >> defect.type;
		outputFile >> defect.name;
		outputFile >> defect.realArea;
		v_defect.push_back(defect);
	}
	outputFile.close();
}
void RJBTestSimple() {
	cv::Mat img_gray;
	vector<CDefect> v_defect;
	getImgDefectParaSample1(img_gray, v_defect);

	CReJudgeBack pp = CReJudgeBack();
	bool result = pp.Process(img_gray, v_defect);
	cout << result << endl;
}

bool RJBTestOne(int serial = 0) {
	CReJudgeBack rjb = CReJudgeBack();
	Mat img = getImgParaSample1();
	vector<CDefect> v_defect;
	getImgDefectParaBySerial(img, v_defect, serial);
	bool result = rjb.Process(img, v_defect);
	cout << result << endl;
	return result;
}

// 全部测试
void RJBTestAll() {
	std::ifstream file("底面二次复判测试用例.csv"); //打开txt文件
	std::ofstream file_result("底面二次复判测试用例_结果.csv"); //打开txt文件
	std::string line;
	int line_num = 1;
	for (int i = 0; i < 4; i++) {
		std::getline(file, line);
		file_result << line << endl;
	}
	for (int i = 0; i < 18; i++) {
		std::getline(file, line);
		char last_char = line.back(); // get the last character of the line
		last_char = last_char == ',' ? line[line.size() - 2] : last_char;
		bool result = RJBTestOne(i);
		std::string line_new = line + to_string(result);
		file_result << line_new << endl;
	}
	file_result.close();
}
