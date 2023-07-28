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

class DLL_EXPORT IYoloSeg
{
public:
    virtual void Init()=0;
    virtual std::vector<Detection> Predict(cv::Mat& img, 
        std::vector<std::vector<Detection>>& res_batch, std::vector<std::vector<std::vector<cv::Point>>>& contours)=0;
    virtual void SetEngineName(std::string engine_name) = 0;
    virtual void SetWeightName(std::string weight_name) = 0;
};
DLL_EXPORT IYoloSeg* createYoloSegInstance();
DLL_EXPORT void get_rect_ex(cv::Mat& img, float bbox[4], cv::Rect& rect);
DLL_EXPORT cv::Rect get_rect(cv::Mat& img, float bbox[4]);
DLL_EXPORT int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names);
