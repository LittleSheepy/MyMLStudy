#include "pch.h"
#include <iostream>
#include <DbgHelp.h>
#include <ctime>

#include "CAlgBase.h"

#ifdef AB_DEBUG
#pragma comment(lib, "DbgHelp.lib")
#endif // AB_DEBUG


#ifdef AB_DEBUG
string AlgBase_img_save_path = "./img_save/";
string AlgBaseBGR = AlgBase_img_save_path + "NumRecBGR/";
string AlgBaseDebug = AlgBase_img_save_path + "NumRecDebug/";
#endif // AB_DEBUG

void CAlgBase::reset() {
    m_debug_imgs.clear();
}

// 获得白色区域 通过轮廓查找
cv::Rect CAlgBase::findWhiteAreaByContour(const cv::Mat& img_gray) {
    sprintf_alg("<<findWhiteArea>> enter findWhiteArea");
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
    sprintf_alg("<<findWhiteArea>> findContours ok");
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
        sprintf_alg("<<findWhiteArea>> findWhiteArea got fiald");
    }

    sprintf_alg("<<findWhiteArea>> result_rect xywh %d %d %d %d", result_rect.x, result_rect.y, result_rect.width, result_rect.height);
    sprintf_alg("<<findWhiteArea>> findWhiteArea out");
    return result_rect;
}
cv::Rect CAlgBase::findWhiteArea(const cv::Mat& img_gray)
{
    double startTime = clock();//计时开始
    cv::Rect result_rect = findWhiteAreaByContour(img_gray);
    cout << "findWhiteAreaByContour: " << clock() - startTime << endl;
    return result_rect;
}

void CAlgBase::sprintf_alg(const char* format, ...) {
    string father_func_name = getFatherFuncName(1);
    string title = "";
    title = "[Alg]{" + father_func_name + "}";
    char buf[256] = { 0 };
    va_list args;
    va_start(args, format);
    sprintf_s(buf, format, args);
    //vsprintf_s(buf, format, args);
    //_vsnprintf_s(buf, sizeof(buf) - 1, format, args);   // 
    va_end(args);
    cout << buf << endl;
    char out_put[1024] = "";
    strcat_s(out_put, title.c_str());
    strcat_s(out_put, buf);
    OutputDebugStringA(out_put);
}

std::string CAlgBase::getFatherFuncName(int n) {
    // 获取堆栈列表
    constexpr int MAX_FRAMES = 64;
    void* frames[MAX_FRAMES];
    int num_frames = CaptureStackBackTrace(0, MAX_FRAMES, frames, nullptr);

    // Initialize the symbol handler
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);

    // Get the symbol name from the address
    char symbolBuffer[sizeof(SYMBOL_INFO) + MAX_SYM_NAME * sizeof(TCHAR)];
    PSYMBOL_INFO symbol = reinterpret_cast<PSYMBOL_INFO>(symbolBuffer);
    symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
    symbol->MaxNameLen = MAX_SYM_NAME;
    // 
    std::string fatherName = "";
    int print_cnt = 0;
    for (int i = 0; i < num_frames; ++i) {
        if (print_cnt >= n) {
            break;
        }
        SymFromAddr(GetCurrentProcess(), (DWORD64)(frames[i]), nullptr, symbol);
        string name = string(symbol->Name);
        if (name == string("CAlgBase::getFatherFuncName") || name == string("CAlgBase::sprintf_alg")) {
            continue;
        }
        fatherName = "[" + string(symbol->Name) + "]" + fatherName;
        print_cnt++;
    }
    return fatherName;
}

string CAlgBase::getTimeString() {
    std::time_t now = std::time(nullptr);
    tm ltm;
    localtime_s(&ltm, &now);
    std::ostringstream oss;
    oss << std::put_time(&ltm, "%Y%m%d%H%M%S");
    oss << std::setfill('0') << std::setw(3) << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() % 1000;
    std::string time_str = oss.str();
    return time_str;
}


