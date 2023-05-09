/*
2023年5月8日
*/
#pragma once
#include <opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "CAlgBase.h"
#include "ReJudgeData.h"
using namespace std;

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
    bool Process_old(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num = 0);
    bool Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num = 0);
    bool defectMatchBox(cv::Mat img, CDefect defect, int serial);
    bool processImg(cv::Mat img, CDefect defect, int serial);
    cv::Mat getMask(vector<cv::Point> points);
    void savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect);
    int getLimit(string bc, int ser);
    vector<vector<CDefect>> groupBBoxes(vector<CDefect> bboxes);
    int groupBBoxes_old(vector<CDefect> bboxes, vector<CDefect>& v_defect1, char bc);
    int HeBing(int serial, char bc);
    bool loadCfg();
public:
    string					        m_className = "【二次复判】";
    vector<CBox>			        m_img1Cfg;
    vector<CBox>			        m_img2Cfg;
    vector<CBox>			        m_img3Cfg;
    vector<CBox>			        m_img4Cfg;
    vector<vector<CBox>>	        m_imgCfg;
    vector<vector<vector<CBox>>>	m_imgCfgAll;
    map<int, vector<CDefect>>	    m_CenterDefectMatched;
    map<int, vector<CDefect>>	    m_BottomDefectMatched;
    map<string, int>		        m_brokenCnt;
    map<string, int>		        m_brokenCfg;
    map<string, vector<int>>	    m_brokenCfg_vect;
    CBoxArray                       m_objs;
    vector<int>                     m_limit_c;
    vector<string>                  m_s_g_c;
    vector<int>                     m_limit_b;
    vector<string>                  m_s_g_b;
    // 模板
    int offset = 0;
    int template_x = 1260;
    cv::Mat img_template;
    string template_path = "template.bmp";
};
