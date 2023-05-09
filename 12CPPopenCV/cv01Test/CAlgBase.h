/*
2023年5月8日
*/
#pragma once
#include <opencv.hpp>
#include "ReJudgeData.h"

//#define AB_DEBUG

#ifdef AB_DEBUG
#endif // AB_DEBUG

using namespace std;
class CAlgBase
{
public:
    string getTimeString();
    cv::Rect findWhiteAreaByContour(const cv::Mat& img_gray);
    cv::Rect findWhiteArea(const cv::Mat& img_gray);
    void sprintf_alg(const char* format, ...);
    void reset();
    void savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, string savePath = "./img_save/AI_para/ReJudgeFront/");
    vector<vector<CDefect>> groupBBoxes(vector<CDefect> bboxes);
    vector<int> getGroupBBoxesWH(vector<CDefect> bboxes);
    vector<int> getGroupBBoxesXYXY(vector<CDefect> bboxes);
    std::map<string, cv::Point> getRowPoint(cv::Mat img_gray, int point_y, int gray_value = 200);
    std::map<string, cv::Point> getColumnPoint(cv::Mat img_gray, int point_x, int gray_value = 200);
    cv::Point getIntersectionPoint(cv::Vec4f Line1, cv::Vec4f Line2);
public:
    string                  m_className = "【算法基类】";
    map<string, cv::Mat>    m_debug_imgs;
};

