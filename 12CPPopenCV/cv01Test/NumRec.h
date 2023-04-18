#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
using namespace cv;
using namespace std;

class CNumRec {
public:
    CNumRec(const std::string template_dir);
    std::vector<int> x_projection(cv::Mat binary_img);
    vector<array<int, 2>> y_projection(cv::Mat binary_img);
    vector<Rect> getNumBoxByProjection(cv::Mat binary_img);
    vector<Rect> getNumBoxByContours(cv::Mat binary_img);
    string processImage(const cv::Mat& img_gray);
    string processImage1(const cv::Mat& img_gray);
    Rect findWhiteArea(const cv::Mat& img_gray);
    Rect findNumArea(const cv::Mat& img_cut);
private:
    std::vector<cv::Mat>    m_template_list;
    cv::Mat                 m_img_white_gray;
};