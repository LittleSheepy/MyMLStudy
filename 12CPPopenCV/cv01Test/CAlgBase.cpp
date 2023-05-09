/*
2023年5月8日
*/
#include "pch.h"
#include <fstream>
#include <iostream>
#include <ctime>
#include <DbgHelp.h>

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
#ifdef AB_DEBUG
    sprintf_alg("[Alg][findWhiteAreaByContour] Enter");
#endif // AB_DEBUG
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

#ifdef AB_DEBUG
    sprintf_alg("[Alg][findWhiteAreaByContour] result_rect xywh %d %d %d %d", result_rect.x, result_rect.y, result_rect.width, result_rect.height);
    sprintf_alg("[Alg][findWhiteAreaByContour] Out");
#endif // AB_DEBUG
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
    string title = "[Alg]";
    char buf[256] = { 0 };
    va_list args;
    va_start(args, format);
    vsprintf_s(buf, format, args);
    va_end(args);
    cout << buf << endl;
    char out_put[1024] = "";
    strcat_s(out_put, title.c_str());
    strcat_s(out_put, buf);
    OutputDebugStringA(out_put);
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

void CAlgBase::savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, string savePath) {
    // 创建或者清空文件夹
    static int num = 1;
    const std::string folder_path = savePath;
    for (int i = 0; i < 4; i++) {
        // 保存图片
        cv::Mat img = v_img[i];
        if (img.empty()) {
            continue;
        }
        std::string filename = folder_path + "/" + std::to_string(num);
        filename = filename + "_" + std::to_string(i);
        std::string img_name = filename + ".jpg";
        cv::imwrite(img_name, img);
        // 保存 vector<CDefect>

        vector<CDefect> v_defect = vv_defect[i];
        std::string vect_name = filename + ".txt";
        std::ofstream outputFile(vect_name);
        // check if the file was successfully opened
        if (outputFile.is_open()) {
            // loop through each element in the vector and write it to the file
            for (const auto& defect : v_defect) {
                //outputFile.write(reinterpret_cast<const char*>(&defect), sizeof(defect));
                outputFile << defect.p1.x << " " << defect.p1.y << endl;
                outputFile << defect.p2.x << " " << defect.p2.y << endl;
                outputFile << defect.area << endl;
                outputFile << defect.type << endl;
                outputFile << defect.name << endl;
            }
            // close the file
            outputFile.close();
        }
        else {
            // handle the case where the file could not be opened
            std::cerr << "Error: could not open file for writing\n";
        }
    }
    num++;
}

// BBOX分组 重叠就分为一组
vector<vector<CDefect>> CAlgBase::groupBBoxes(vector<CDefect> bboxes) {
    vector<vector<CDefect>> groups;
    for (int i = 0; i < bboxes.size(); i++) {
        bool added = false;
        for (int j = 0; j < groups.size(); j++) {
            for (int k = 0; k < groups[j].size(); k++) {
                //if (overlap(bboxes[i], groups[j][k])) {
                if (bboxes[i].overlap(groups[j][k])) {
                    groups[j].push_back(bboxes[i]);
                    added = true;
                    break;
                }
            }
            if (added) {
                // Merge groups that overlap with each other
                for (int k = j + 1; k < groups.size(); k++) {
                    bool overlapFound = false;
                    for (int l = 0; l < groups[k].size(); l++) {
                        if (bboxes[i].overlap(groups[k][l])) {
                            overlapFound = true;
                            break;
                        }
                    }
                    if (overlapFound) {
                        groups[j].insert(groups[j].end(), groups[k].begin(), groups[k].end());
                        groups.erase(groups.begin() + k);
                        k--;
                    }
                }
                break;
            }
        }
        if (!added) {
            groups.push_back({ bboxes[i] });
        }
    }
    return groups;
}


// 获得一组bbox的WH
vector<int> CAlgBase::getGroupBBoxesWH(vector<CDefect> bboxes) {
    vector<int> resultWH;
    int x_min = 5000, x_max = 0, y_min = 5000, y_max = 0;
    for (int k = 0; k < bboxes.size(); k++) {
        CDefect v = bboxes[k];
        if (min(v.p1.x, v.p2.x) < x_min) {
            x_min = min(v.p1.x, v.p2.x);
        }
        if (min(v.p1.y, v.p2.y) < y_min) {
            y_min = min(v.p1.y, v.p2.y);
        }
        if (max(v.p1.x, v.p2.x) > x_max) {
            x_max = max(v.p1.x, v.p2.x);
        }
        if (max(v.p1.y, v.p2.y) > y_max) {
            y_max = max(v.p1.y, v.p2.y);
        }
    }

    int defect_w = x_max - x_min;
    int defect_h = y_max - y_min;
    resultWH.push_back(defect_w);
    resultWH.push_back(defect_h);
    sprintf_alg("[getGroupBBoxesWH] defect_w=%d, defect_h=%d", defect_w, defect_h);
    return resultWH;
}

// 获得一组bbox的WH
vector<int> CAlgBase::getGroupBBoxesXYXY(vector<CDefect> bboxes) {
    vector<int> resultXYXY;
    int x_min = 5000, x_max = 0, y_min = 5000, y_max = 0;
    for (int k = 0; k < bboxes.size(); k++) {
        CDefect v = bboxes[k];
        if (min(v.p1.x, v.p2.x) < x_min) {
            x_min = min(v.p1.x, v.p2.x);
        }
        if (min(v.p1.y, v.p2.y) < y_min) {
            y_min = min(v.p1.y, v.p2.y);
        }
        if (max(v.p1.x, v.p2.x) > x_max) {
            x_max = max(v.p1.x, v.p2.x);
        }
        if (max(v.p1.y, v.p2.y) > y_max) {
            y_max = max(v.p1.y, v.p2.y);
        }
    }

    resultXYXY.push_back(x_min);
    resultXYXY.push_back(y_min);
    resultXYXY.push_back(x_max);
    resultXYXY.push_back(y_max);
    return resultXYXY;
}

// 遍历行
std::map<string, cv::Point> CAlgBase::getRowPoint(cv::Mat img_gray, int point_y, int gray_value) {
    int w = img_gray.cols;
    int x = 0;
    std::map<cv::String, cv::Point> RowPoints;
    for (int point_x = 0; point_x < w; point_x++) {
        if (img_gray.at<uchar>(point_y, point_x) > gray_value) {
            int y = point_x;
            RowPoints["whiteleft"] = cv::Point(point_x, point_y);
            break;
        }
    }
    for (int point_x = x + 1; point_x < w; point_x++) {
        if (img_gray.at<uchar>(point_y, point_x) < gray_value) {
            RowPoints["whiteright"] = cv::Point(point_x, point_y);
            break;
        }
    }
    return RowPoints;
}

map<string, cv::Point> CAlgBase::getColumnPoint(cv::Mat img_gray, int point_x, int gray_value) {
    int h = img_gray.rows;
    int y = 0;
    map<string, cv::Point> EarPoints;

    for (int point_y = 0; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) > gray_value) {
            y = point_y;
            EarPoints["whitetop"] = { point_x, point_y };
            break;
        }
    }

    for (int point_y = y + 1; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) < gray_value) {
            y = point_y;
            EarPoints["whitebottom"] = { point_x, point_y };
            break;
        }
    }

    return EarPoints;
}

cv::Point CAlgBase::getIntersectionPoint(cv::Vec4f line1, cv::Vec4f line2) {
    // Define variables for slope and y-intercepts of line1 and line2
    float slope1, slope2, yIntercept1, yIntercept2;
    // Compute slope and y-intercept of line1
    slope1 = line1[1] / line1[0];
    yIntercept1 = line1[3] - slope1 * line1[2];
    // Compute slope and y-intercept of line2
    slope2 = line2[1] / line2[0];
    yIntercept2 = line2[3] - slope2 * line2[2];

    // Find the intersection point of the two lines
    double x = (yIntercept2 - yIntercept1) / (slope1 - slope2 + 0.000001);
    double y = slope1 * x + yIntercept1;
    return cv::Point((int)x, (int)y);
}