/*
2023年4月28日
*/
#pragma once
#include <opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "CAlgBase.h"
using namespace std;

// 瑕疵结构体
struct CDefect
{
    cv::Point   p1;	        // 左上点
    cv::Point   p2;	        // 右下点
    int         area;	    // 缺陷面积
    int	        type;	    // 缺陷类型 11破损、12毛边
    string	    name;	    // 缺陷类型 11破损、12毛边
    CDefect() {}
    CDefect(cv::Point p1, cv::Point p2, int area, int type = 11, string name = "") :p1(p1), p2(p2), area(area), type(type), name(name) {}
};

// 瑕疵结构体
struct CArrayCfg
{
    int   cnt;	// 允许缺陷个数
    int	  area;	// 允许缺陷面积
    CArrayCfg() :cnt(0), area(0) {}
    CArrayCfg(int cnt, int area) :cnt(cnt), area(area) {}
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
    int	w;
    int	h;
    int n_defect;
    bool state;
    CBox(string name, string arr_name, int serial, cv::Point p1, cv::Point p2, int	area = 0, int w = 0, int h = 0) :
        name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), area(area), w(w), h(h), n_defect(0), state(true) {}
};
class CBoxArray {
public:
    vector<CBox> v_obj;
    int state;
};

class CPostProcessor :CAlgBase {
public:
    CPostProcessor();
    void imgCfgInit();
    void imgCfgInitByOffSet();
    void imgCfgInitByOffSet2();
    map<string, cv::Point> getRowWhitePoint(const cv::Mat& img_gray, int point_y);
    cv::Point findWhiteAreaByTemplate(const cv::Mat& img_bgr);
    //cv::Rect findWhiteArea(const cv::Mat& img_bgr);
    void setOffSet(cv::Mat img_bgr, int camera_num = 0);
    // v_img 四张图片
    // vv_defect 四个CDefect缺陷列表 
    bool Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num = 0);
    bool processImg(cv::Mat img, CDefect defect, int serial);
    cv::Mat getMask(vector<cv::Point> points);
    void savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect);
public:
    string					m_className = "【二次复判】";
    vector<CBox>			m_img1Cfg;
    vector<CBox>			m_img2Cfg;
    vector<CBox>			m_img3Cfg;
    vector<CBox>			m_img4Cfg;
    vector<vector<CBox>>	m_imgCfg;
    map<string, int>		m_brokenCnt;
    map<string, int>		m_brokenCfg;
    CBoxArray               m_objs;
    // 模板
    int offset = 0;
    int template_x = 1260;
    cv::Mat img_template;
    string template_path = "template.bmp";
};
