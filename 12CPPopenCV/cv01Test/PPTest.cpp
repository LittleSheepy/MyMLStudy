#include <iostream>
#include<ctime>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>

#include "CPostProcessor.h"
using namespace std;
using namespace cv;

vector<Mat> getImgParaSample() {
	vector<Mat> v_img;
	string dir_root = "D:/04DataSets/ningjingLG/all/";
	string img_first_name = "black_0074690_CM1_";
	for (int i = 0; i < 4; i++) {
		string img_path = dir_root + img_first_name + to_string(i + 1) + ".bmp";
		v_img.push_back(imread(img_path, cv::IMREAD_COLOR));
	}
	return v_img;
}

void getDefectParaSample(vector<Mat>& v_img, vector<vector<CDefect>>& vv_defect) {
	v_img = getImgParaSample();
	vector<CDefect> v_defect = { {{800,800},{900,900}, 6, 1 } };
	vv_defect.push_back(v_defect);
	vv_defect.push_back({});
	vv_defect.push_back({});
	vv_defect.push_back({});
}

void getImgDefectParaBySerial(vector<Mat>& v_img, vector<vector<CDefect>>& vv_defect, int serial=1) {
	const std::string folder_path = "./AI_para042914";
	for (int i = 0; i < 4; i++) {
		// 读取图片
		/*Mat img = v_img[i];*/
		std::string filename = folder_path + "/";
		filename = filename + std::to_string(serial) + "_" + std::to_string(i);
		std::string img_name = filename + ".bmp";
		v_img.push_back(imread(img_name));

		// 读取 vector<CDefect>
		vector<CDefect> v_defect;
		std::string vect_name = filename + ".txt";
		std::ifstream outputFile(vect_name, std::ios::binary);
		CDefect defect;
		while (!outputFile.eof()) {
			outputFile >> defect.p1.x >> defect.p1.y;
			outputFile >> defect.p2.x >> defect.p2.y;
			outputFile >> defect.area;
			outputFile >> defect.type;
			outputFile >> defect.name;
			v_defect.push_back(defect);
		}
		vv_defect.push_back(v_defect);
		outputFile.close();
	}
}
void PPTestOne(int serial = 1) {
	CPostProcessor pp = CPostProcessor();
	vector<Mat> v_img;
	vector<vector<CDefect>> vv_defect;
	getImgDefectParaBySerial(v_img, vv_defect, serial);
	bool result = pp.Process(v_img, vv_defect);
	cout << result << endl;
}
// 全部测试
void PPTestAll() {
	for (int i = 0; i < 4; i++) {
		PPTestOne(i+1);
	}
}