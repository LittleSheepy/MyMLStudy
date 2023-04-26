#include "pch.h"
#include <iostream>
#include <Windows.h>

#include "CAlgBase.h"



#ifdef AB_DEBUG
string AlgBase_img_save_path = "./img_save/";
string AlgBaseBGR = AlgBase_img_save_path + "NumRecBGR/";
string AlgBaseDebug = AlgBase_img_save_path + "NumRecDebug/";
#endif // AB_DEBUG
cv::Rect CAlgBase::findWhiteArea(const cv::Mat& img_gray) {
    sprintf("<<findWhiteArea>> enter findWhiteArea");
    cv::Mat img_gray_binary;
    cv::threshold(img_gray, img_gray_binary, 250, 255, cv::THRESH_BINARY);
#ifdef AB_DEBUG
    static int index;
    string file_name_debug = AlgBaseDebug + to_string(index) + ".bmp";
    cv::imwrite(file_name_debug, img_gray);
    m_debug_imgs["findWhiteArea_img_gray_binary"] = img_gray_binary;
#endif // AB_DEBUG

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_gray_binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

#ifdef AB_DEBUG
    sprintf("<<findWhiteArea>> findContours ok");
    cv::Mat img_bgr_contours;
    cv::cvtColor(img_gray, img_bgr_contours, cv::COLOR_GRAY2BGR);
    drawContours(img_bgr_contours, contours, -1, cv::Scalar(0, 0, 255), 2);
    m_debug_imgs["findWhiteArea_contours"] = img_bgr_contours;
#endif // AB_DEBUG
    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect(contours_poly[i]);
    }
    // Filter out contours with an area less than 100
    vector<cv::Rect> filteredRects;
    for (cv::Rect rect : boundRect)
    {
        if (rect.area() >= 100000 && rect.area() < 450000)
        {
            filteredRects.push_back(rect);
        }
    }
    cv::Rect result_rect = cv::Rect(0, 0, 0, 0);

    if (filteredRects.size() > 0) {
        result_rect = filteredRects[0];
    }
    else {
        sprintf("<<findWhiteArea>> findWhiteArea got fiald");
    }

    sprintf("<<findWhiteArea>> result_rect xywh %d %d %d %d", result_rect.x, result_rect.y, result_rect.width, result_rect.height);
    sprintf("<<findWhiteArea>> findWhiteArea out");
    return result_rect;
}
void CAlgBase::sprintf(const char* format, ...) {
    char buf[125];
    va_list args;
    va_start(args, format);
    vsprintf_s(buf, format, args);
    va_end(args);
    OutputDebugStringA(buf);
}