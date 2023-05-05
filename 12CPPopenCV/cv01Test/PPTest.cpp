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

void getImgDefectParaBySerial(vector<Mat>& v_img, vector<vector<CDefect>>& vv_defect, int serial = 1) {
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
			outputFile >> defect.p1.x;
			if (outputFile.eof()) {
				break;
			}
			outputFile >> defect.p1.y;
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

void getImgDefectParaBySerial2(vector<Mat>& v_img, vector<vector<CDefect>>& vv_defect, int serial = 1) {
	const std::string folder_path = "./AI_para_test";
	for (int i = 0; i < 4; i++) {
		// 读取图片
		/*Mat img = v_img[i];*/
		std::string filename = folder_path + "/";
		std::string img_name = filename + std::to_string(serial%2) + "_" + std::to_string(i) + ".bmp";
		v_img.push_back(imread(img_name));

		// 读取 vector<CDefect>
		vector<CDefect> v_defect;
		std::string vect_name = filename + std::to_string(serial) + "_" + std::to_string(i) + ".txt";
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
			v_defect.push_back(defect);
		}
		vv_defect.push_back(v_defect);
		outputFile.close();
	}
}
bool PPTestOne(int serial = 0) {
	CPostProcessor pp = CPostProcessor();
	vector<Mat> v_img;
	vector<vector<CDefect>> vv_defect;
	getImgDefectParaBySerial2(v_img, vv_defect, serial);
	bool result = pp.Process(v_img, vv_defect, serial % 2);
	cout << result << endl;
	return result;
}
// 全部测试
void PPTestAll() {
	std::ifstream file("二次复判测试用例.csv"); //打开txt文件
	std::ofstream file_result("二次复判测试用例_结果.csv"); //打开txt文件
	std::string line;
	int line_num = 1;
	for (int i = 0; i < 4; i++) {
		std::getline(file, line);
		file_result << line << endl;
	}
	for (int i = 0; i < 16; i++) {
		std::getline(file, line);
		char last_char = line.back(); // get the last character of the line
		last_char = last_char == ',' ? line[line.size() - 2] : last_char;
		bool result = PPTestOne(i);
		std::string line_new = line + to_string(result);
		file_result << line_new << endl;
	}
	file_result.close();
}