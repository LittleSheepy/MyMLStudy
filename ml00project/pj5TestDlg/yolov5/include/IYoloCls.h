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

class DLL_EXPORT IYoloCls
{
public:
    virtual void Init()=0;
    virtual void Serialize()=0;
    virtual int Predict(cv::Mat& img)=0;
    virtual void SetEngineName(std::string engine_name) = 0;
    virtual void SetWeightName(std::string weight_name) = 0;
};
DLL_EXPORT IYoloCls* createYoloClsInstance();
DLL_EXPORT int read_files_in_dir(const char* p_dir_name, std::vector<std::string>& file_names);
