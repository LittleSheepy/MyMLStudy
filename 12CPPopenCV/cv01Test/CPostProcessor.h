#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
using namespace std;
using namespace cv;

class CBox
{
public:
	string name;
	string arr_name;
	int serial;
	Point p1;
	Point p2;
	bool state;
	CBox() {};
	CBox(string name, string arr_name, int serial, Point p1, Point p2) :name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), state(true){}
};
class CBoxArray {
public:
	vector<CBox> v_obj;
	int state;
};

class CPostProcessor
{
public:
	CPostProcessor();
	bool Process(vector<Mat> v_img);
	void processImg1(Mat img, int serial);
public:
	vector<CBox> m_img1Cfg;
	vector<CBox> m_img2Cfg;
	vector<CBox> m_img3Cfg;
	vector<CBox> m_img4Cfg;
	map<string, int>		m_brokenCnt;
	CBoxArray m_objs;
};

