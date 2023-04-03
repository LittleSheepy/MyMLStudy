#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <map>
using namespace std;
using namespace cv;
// 覴ýṹ��
struct CDefect
{
	Point p1;	// ���ϵ�
	Point p2;	// ���µ�
	int   area;	// ȱ�����
	int	  type;	// ȱ������ ����ë��
	CDefect(Point p1, Point p2, int area, int type):p1(p1), p2(p2), area(area), type(type){}
};


class CBox
{
public:
	string name;
	string arr_name;
	int serial;
	Point p1;
	Point p2;
	int n_defect;
	bool state;
	CBox(string name, string arr_name, int serial, Point p1, Point p2) :
		name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), n_defect(0), state(true) {}
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
	// v_img ����ͼƬ
	// vv_defect �ĸ�CDefectȱ���б�
	bool Process(vector<Mat> v_img, vector<vector<CDefect>> vv_defect);
	void processImg1(Mat img, CDefect defect, int serial);
	Mat getMask(vector<Point> points);
public:
	vector<CBox> m_img1Cfg;
	vector<CBox> m_img2Cfg;
	vector<CBox> m_img3Cfg;
	vector<CBox> m_img4Cfg;
	map<string, int>		m_brokenCnt;
	map<string, int>		m_brokenCfg;
	CBoxArray m_objs;
};

