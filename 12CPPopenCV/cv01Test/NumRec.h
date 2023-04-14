#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

class CNumRec {
public:
    CNumRec(const std::string template_dir);
    void processImage(const cv::Mat& img_gray);
    Rect findWhiteArea(const cv::Mat& img_gray);
private:
    std::vector<cv::Mat> template_list;
    cv::Mat img_white_gray;
};