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
	CDefect(Point p1, Point p2, int area, int type) :p1(p1), p2(p2), area(area), type(type) {}
};

// 覴ýṹ��
struct CArrayCfg
{
	int   cnt;	// ����ȱ�ݸ���
	int	  area;	// ����ȱ�����
	CArrayCfg() :cnt(0), area(0) {}
	CArrayCfg(int type, int area) :cnt(cnt), area(area) {}
};

class CBox
{
public:
	string name;
	string arr_name;
	int serial;
	Point p1;
	Point p2;
	int	area;
	int n_defect;
	bool state;
	CBox(string name, string arr_name, int serial, Point p1, Point p2, int	area=0) :
		name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), area(area), n_defect(0), state(true) {}
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
	void imgCfgInit();
	void imgCfgInitByOffSet();
	cv::Point findWhiteArea(cv::Mat img_bgr);
	void setOffSet(cv::Mat img_bgr);
	// v_img ����ͼƬ
	// vv_defect �ĸ�CDefectȱ���б�
	bool Process(vector<Mat> v_img, vector<vector<CDefect>> vv_defect);
	void processImg(Mat img, CDefect defect, int serial);
	Mat getMask(vector<Point> points);
	void savePara(vector<Mat> v_img, vector<vector<CDefect>> vv_defect);
public:
	vector<CBox>			m_img1Cfg;
	vector<CBox>			m_img2Cfg;
	vector<CBox>			m_img3Cfg;
	vector<CBox>			m_img4Cfg;
	vector<vector<CBox>>	m_imgCfg;
	map<string, int>		m_brokenCnt;
	map<string, int>		m_brokenCfg;
	CBoxArray m_objs;
	// ģ��
	int offset = 0;
	int template_x = 1260;
	Mat img_template;
	string template_path = "template.bmp";
};

