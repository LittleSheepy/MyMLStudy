// 遍历图片
#include "pch.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>
#include <fstream>
#include <string>
#include "CPostProcessor.h"
#include "CAlgBase.h"
#include "CNumRec.h"

// opencv
//#include <opencv2/core/utils/logger.hpp>
//using namespace cv::utils::logging;
////setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
//cv::utils::logging::setLogLevel(0);

using namespace std;
using namespace cv;
void FolderTest();
void NumRecTest();
bool PPTestOne(int serial = 1);
void PPTestAll();
void saveJson();

void PP_test()
{
	string dir_root = "D:/04DataSets/ningjingLG/all/";
	string img_first_name = "black_0074690_CM1_";
	vector<Mat> v_img;
	//for (int i = 0; i < 4; i++) {
	//	string img_path = dir_root + img_first_name + to_string(i + 1) + ".bmp";
	//	//v_img.push_back(imread(img_path, cv::IMREAD_GRAYSCALE));
	//	v_img.push_back(imread(img_path));
	//}
	CPostProcessor pp = CPostProcessor();
	vector<vector<CDefect>> vv_defect;
	//vector<CDefect> v_defect = { {{800,800},{900,900}, 6, 1 } };
	//vv_defect.push_back(v_defect);
	//vv_defect.push_back({});
	//vv_defect.push_back({});
	//vv_defect.push_back({});

	const std::string folder_path = "./AI_para";
	for (int i = 0; i < 4; i++) {
		// 读取图片
		/*Mat img = v_img[i];*/
		std::string filename = folder_path + "/" + std::to_string(1);
		filename = filename + "_" + std::to_string(i);
		std::string img_name = filename + ".bmp";
		v_img.push_back(imread(img_name));

		// 读取 vector<CDefect>
		vector<CDefect> v_defect;
		std::string vect_name = filename + ".bin";
		std::ifstream outputFile(vect_name, std::ios::binary);
		CDefect defect;
		while (outputFile.read(reinterpret_cast<char*>(&defect), sizeof(CDefect)))
		{
			v_defect.push_back(defect);
		}
		vv_defect.push_back(v_defect);
		outputFile.close();
	}

	bool result = pp.Process(v_img, vv_defect);
	result = pp.Process(v_img, vv_defect);
	result = pp.Process(v_img, vv_defect);
	cout << result << endl;
}
void ABTest() {
	CAlgBase ab_base = CAlgBase();
	string time_str = ab_base.getTimeString();
	ab_base.sprintf_alg(time_str.c_str());
}
int main()
{
	SetConsoleOutputCP(65001);
	//ABTest();

	//PPtest();
	saveJson();
	//NumRecTest();
	//FolderTest();
	return 0;
}