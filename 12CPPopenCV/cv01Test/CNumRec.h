#pragma once
#include <opencv.hpp>
#include <string>
#include <vector>
#include "CAlgBase.h"
using namespace std;
// 参数
struct CNumRecPara
{
    cv::Mat img_bgr;
    string str_result;
    string str_pre_saved;
};

class CNumRec:CAlgBase {
public:
    CNumRec();
    CNumRec(const std::string template_dir);
    bool processImage(CNumRecPara& num_rec_para);
    bool processImage(const cv::Mat& img_gray, string& str_result);
    string processImage(const cv::Mat& img_gray);
private:
    std::vector<int> x_projection(cv::Mat binary_img);
    vector<array<int, 2>> y_projection(cv::Mat binary_img);
    vector<cv::Rect> getNumBoxByProjection(cv::Mat binary_img);
    vector<cv::Rect> getNumBoxByContours(cv::Mat binary_img);
    string processImage1(const cv::Mat& img_gray);
    cv::Rect findWhiteArea(const cv::Mat& img_gray);
    cv::Rect findNumArea(const cv::Mat& img_cut);
    std::vector<cv::Mat>    m_template_list;
    cv::Mat                 m_img_white_gray;
};