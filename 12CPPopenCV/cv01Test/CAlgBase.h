#pragma once
#include <opencv.hpp>

#define AB_DEBUG

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
    std::string getFatherFuncName(int n = 2);
    void reset();
public:
    string                  m_className = "【算法基类】";
    map<string, cv::Mat>    m_debug_imgs;
};

