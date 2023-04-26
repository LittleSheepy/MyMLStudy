#pragma once
#include <opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "CAlgBase.h"
using namespace std;

// 覴ýṹ��
struct CDefect
{
	cv::Point p1;	// ���ϵ�
	cv::Point p2;	// ���µ�
	int   area;	// ȱ�����
	int	  type;	// ȱ������ ����ë��
	CDefect() {}
	CDefect(cv::Point p1, cv::Point p2, int area, int type) :p1(p1), p2(p2), area(area), type(type) {}
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
	cv::Point p1;
	cv::Point p2;
	int	area;
	int n_defect;
	bool state;
	CBox(string name, string arr_name, int serial, cv::Point p1, cv::Point p2, int	area = 0) :
		name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), area(area), n_defect(0), state(true) {}
};
class CBoxArray {
public:
	vector<CBox> v_obj;
	int state;
};

class CPostProcessor :CAlgBase
{
public:
	CPostProcessor();
	void imgCfgInit();
	void imgCfgInitByOffSet();
	void imgCfgInitByOffSet2();
	map<string, cv::Point> getRowWhitePoint(const cv::Mat& img_gray, int point_y);
	cv::Point findWhiteAreaByTemplate(const cv::Mat& img_bgr);
	//cv::Rect findWhiteArea(const cv::Mat& img_bgr);
	void setOffSet(cv::Mat img_bgr, int camera_num = 0);
	// v_img ����ͼƬ
	// vv_defect �ĸ�CDefectȱ���б� 
	bool Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num = 0);
	void processImg(cv::Mat img, CDefect defect, int serial);
	cv::Mat getMask(vector<cv::Point> points);
	void savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect);
public:
	string					m_className = "�����θ��С�";
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
	cv::Mat img_template;
	string template_path = "template.bmp";
};

