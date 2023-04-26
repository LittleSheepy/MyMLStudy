#pragma once
#include <opencv.hpp>

#define AB_DEBUG 0

using namespace std;
class CAlgBase
{
public:
    cv::Rect findWhiteArea(const cv::Mat& img_gray);
#ifdef AB_DEBUG
#endif // AB_DEBUG
    map<string, cv::Mat> m_debug_imgs;
    void sprintf(const char* format, ...);

};

