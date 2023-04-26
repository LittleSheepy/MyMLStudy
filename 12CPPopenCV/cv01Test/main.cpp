// 遍历图片
#include <Windows.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>
#include "CPostProcessor.h"
#include "quickopencv.h"
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <DbgHelp.h>
using namespace std;
using namespace cv;
void TraversalPicture();
void NumRecTest();
void cvtest()
{
	std::cout << "Hello World!\n";
	string imagename = "E:/23_03_24_opencv_build/opencv450/butterfly.png"; 
	Mat img = imread(imagename, IMREAD_COLOR);	//彩色
	//Mat img = imread(imagename, IMREAD_GRAYSCALE);	//黑白

	//如果读入图像失败
	if (img.empty())
	{
		fprintf(stderr, "Can not load image %s\n", imagename.c_str());
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

void PPtest()
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
void BB();
void BB() {
	constexpr int MAX_FRAMES = 64;
	void* frames[MAX_FRAMES];
	int num_frames = CaptureStackBackTrace(0, MAX_FRAMES, frames, nullptr);

	SYMBOL_INFO* symbol = (SYMBOL_INFO*)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
	symbol->MaxNameLen = 255;
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
	std::cout << "Stack trace:\n";
	for (int i = 0; i < num_frames; ++i) {
		if (SymFromAddr(GetCurrentProcess(), (DWORD64)(frames[i]), 0, symbol)) {
			std::cerr << symbol->Name << '\n';
		}
		else {
			std::cerr << "Failed to get symbol information for address " << frames[i] << '\n';
		}
	}
	free(symbol);
}
int CC() {
	constexpr int MAX_FRAMES = 64;
	void* frames[MAX_FRAMES];
	int num_frames = CaptureStackBackTrace(0, MAX_FRAMES, frames, nullptr);
	// Define a function pointer to a function with no arguments and no return value
	void (*funcPtr)() = []() {
		std::cout << "Hello World!" << std::endl;
	};

	// Get the address of the function pointer
	DWORD_PTR funcAddr = reinterpret_cast<DWORD_PTR>(funcPtr);
	// Initialize the symbol handler
	SymInitialize(GetCurrentProcess(), nullptr, TRUE);

	// Get the symbol name from the address
	char symbolBuffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
	PSYMBOL_INFO symbol = reinterpret_cast<PSYMBOL_INFO>(symbolBuffer);
	symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
	symbol->MaxNameLen = MAX_SYM_NAME;
	for (int i = 0; i < num_frames; ++i) {

		if (SymFromAddr(GetCurrentProcess(), (DWORD64)(frames[i]), nullptr, symbol)) {
			std::cout << "Symbol name: " << symbol->Name << std::endl;
		}
		else {
			std::cerr << "Failed to get symbol name" << std::endl;
		}
	}
	// Clean up the symbol handler
	SymCleanup(GetCurrentProcess());

	return 0;
}
void A() {
	CC();
}
int main()
{
	A();
	//NumRecTest();
	//PPtest();
	return 0;
}
