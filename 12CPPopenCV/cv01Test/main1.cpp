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

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;
void FolderTest();
//void TraversalPicture();
void NumRecTest();
void PPTestAll();
bool PPTestOne(int serial = 0);
void RJFTestSimple();
void RJBTestSimple();
void RJBTestAll();
bool RJBTestOne(int serial = 0);

void img_cvtTest(void);
void PPtest()
{
	string dir_root = "D:/04DataSets/05nanjingLG/01CeMian/all/";
	string img_first_name = "black_0074690_CM1_";
	CPostProcessor pp = CPostProcessor();
	vector<vector<CDefect>> vv_defect;
	vector<Mat> v_img;

	//for (int i = 0; i < 4; i++) {
	//	string img_path = dir_root + img_first_name + to_string(i + 1) + ".bmp";
	//	//v_img.push_back(imread(img_path, cv::IMREAD_GRAYSCALE));
	//	v_img.push_back(imread(img_path));
	//}
	//vector<CDefect> v_defect = { {{800,800},{900,900}, 6, 11} };
	//vv_defect.push_back(v_defect);
	//vv_defect.push_back({});
	//vv_defect.push_back({});
	//vv_defect.push_back({});

	const std::string folder_path = "./AI_para042914";
	for (int i = 0; i < 4; i++) {
		// 读取图片
		/*Mat img = v_img[i];*/
		std::string filename = folder_path + "/";	//  + std::to_string(1)
		filename = filename + "1_" + std::to_string(i);
		std::string img_name = filename + ".bmp";
		v_img.push_back(imread(img_name));

		// 读取 vector<CDefect>
		vector<CDefect> v_defect;
		std::string vect_name = filename + ".txt";
		std::ifstream outputFile(vect_name, std::ios::binary);
		CDefect defect;
		//while (outputFile.read(reinterpret_cast<char*>(&defect), sizeof(CDefect)))
		//{
		//	v_defect.push_back(defect);
		//}
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

	bool result = pp.Process(v_img, vv_defect);
	result = pp.Process(v_img, vv_defect);
	result = pp.Process(v_img, vv_defect);
	std::cout << result << endl;
}
void PPtest1()
{
	string dir_root = "D:/04DataSets/05nanjingLG/01CeMian/all/";
	string img_first_name = "black_0074690_CM1_";
	vector<Mat> v_img;
	for (int i = 0; i < 4; i++) {
		string img_path = dir_root + img_first_name + to_string(i + 1) + ".bmp";
		//v_img.push_back(imread(img_path, cv::IMREAD_GRAYSCALE));
		v_img.push_back(imread(img_path));
	}
	CPostProcessor pp = CPostProcessor();
	vector<vector<CDefect>> vv_defect;
	//vector<CDefect> v_defect = { {{450,700},{700,1630}, 6, 1 } };
	//vv_defect.push_back(v_defect);
	//vv_defect.push_back(v_defect);
	//vv_defect.push_back(v_defect);
	//vv_defect.push_back(v_defect);
	//std::ofstream writeFile("v_defect.bin", std::ios::binary);
	//// Write the vector to the file using file.write()
	//writeFile.write(reinterpret_cast<char*>(&v_defect[0]), v_defect.size() * sizeof(CDefect));
	//// Close the file after writing
	//writeFile.close();

	//std::vector<CDefect> read_defects;
	//std::ifstream infile("D:/04DataSets/ningjingLG/AI_para/2_3.bin", std::ios::binary);
	//CDefect temp_defect;
	//while (infile.read(reinterpret_cast<char*>(&temp_defect), sizeof(CDefect))) {
	//	read_defects.push_back(temp_defect);
	//}
	bool success = pp.Process(v_img, vv_defect);
	std::cout << success << endl;
	pp.Process(v_img, vv_defect);
}
void ABTest() {
	CAlgBase ab_base = CAlgBase();
	string time_str = ab_base.getTimeString();
	ab_base.sprintf_alg(time_str.c_str());
}


///// bbox 分组
struct BBox {
	int x1, y1, x2, y2;

	//bool operator==(const BBox& other) const {
	//	return x1 == other.x1 && y1 == other.y1 && x2 == other.x2 && y2 == other.y2;
	//}

	//bool operator<(const BBox& other) const {
	//	if (x1 != other.x1) {
	//		return x1 < other.x1;
	//	}
	//	if (y1 != other.y1) {
	//		return y1 < other.y1;
	//	}
	//	if (x2 != other.x2) {
	//		return x2 < other.x2;
	//	}
	//	return y2 < other.y2;
	//}
};

bool overlap(BBox b1, BBox b2) {
	return (b1.x1 <= b2.x2 && b1.x2 >= b2.x1 && b1.y1 <= b2.y2 && b1.y2 >= b2.y1);
}
vector<vector<BBox>> groupBBoxes(vector<BBox> bboxes) {
	vector<vector<BBox>> groups;
	for (int i = 0; i < bboxes.size(); i++) {
		bool added = false;
		for (int j = 0; j < groups.size(); j++) {
			for (int k = 0; k < groups[j].size(); k++) {
				if (overlap(bboxes[i], groups[j][k])) {
					groups[j].push_back(bboxes[i]);
					added = true;
					break;
				}
			}
			if (added) {
				// Merge groups that overlap with each other
				for (int k = j + 1; k < groups.size(); k++) {
					bool overlapFound = false;
					for (int l = 0; l < groups[k].size(); l++) {
						if (overlap(bboxes[i], groups[k][l])) {
							overlapFound = true;
							break;
						}
					}
					if (overlapFound) {
						groups[j].insert(groups[j].end(), groups[k].begin(), groups[k].end());
						groups.erase(groups.begin() + k);
						k--;
					}
				}
				break;
			}
		}
		if (!added) {
			groups.push_back({ bboxes[i] });
		}
	}
	return groups;
}
int groupBBoxesTest() {
	vector<BBox> bboxes = { {1, 1, 3, 3}, {4, 4, 6, 6}, {7,7,9,9}, {2, 2, 8, 8} };
	vector<vector<BBox>> groups = groupBBoxes(bboxes);
	for (auto group : groups) {
		for (auto bbox : group) {
			cout << "(" << bbox.x1 << "," << bbox.y1 << "," << bbox.x2 << "," << bbox.y2 << ") ";
		}
		cout << endl;
	}
	return 0;
}


int main()
{
	SetConsoleOutputCP(65001);
	//ABTest();
	//groupBBoxesTest();
	//RJBTestSimple();
	//RJBTestOne(18);
	//RJBTestAll();
	//PPTestAll();
	//RJFTestSimple();
	//NumRecTest();
	//FolderTest();
	img_cvtTest();
	return 0;
}