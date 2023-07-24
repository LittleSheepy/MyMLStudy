#pragma once

/*
用例:
// 1.创建YoloDet类对象
IYoloDet* yolo_det = createYoloDetInstance();
// 2. 初始化类对象
yolo_det->Init();
*/

#include <opencv2/opencv.hpp>
#include "types.h"
#include "config.h"

class DLL_EXPORT IYoloDet
{
public:
    virtual void Init()=0;
    virtual std::vector<Detection> Predict(cv::Mat& img)=0;
};
DLL_EXPORT IYoloDet* createYoloDetInstance();
DLL_EXPORT void get_rect_ex(cv::Mat& img, float bbox[4], cv::Rect& rect);
DLL_EXPORT cv::Rect get_rect(cv::Mat& img, float bbox[4]);
DLL_EXPORT int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names);
