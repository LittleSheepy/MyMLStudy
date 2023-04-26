// NumRec.cpp
#include "pch.h"
#include <iostream>
#include <numeric>
#include <Windows.h>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <map>

#include "CNumRec.h"


#define PP_DEBUG 0

#ifdef PP_DEBUG
//#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
//#include <experimental/filesystem> // C++14标准引入的文件系统库
//#include <sys/stat.h>
//namespace fs = std::experimental::filesystem;
//string img_save_path_str = "D:/00myGitHub/00MyMLStudy/12CPPopenCV/bin/img_save/";
string img_save_path = "./img_save/";
string NumRecBGR = img_save_path + "NumRecBGR/";
string NumRecDebug = img_save_path + "NumRecDebug/";
map<string, cv::Mat> map_img;
#endif // PP_DEBUG


CNumRec::CNumRec() {
    string template_dir = "./Template/";
    // Load templates
    for (int i = 0; i < 10; i++) {
        cv::Mat template_img = cv::imread(template_dir + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);
        //Mat binary_img_i;
        //cv::threshold(template_img, binary_img_i, 250, 255, THRESH_BINARY_INV);
        m_template_list.push_back(template_img);
    }
    cv::String white_template_path = template_dir + "white_template1.bmp";
    m_img_white_gray = cv::imread(white_template_path, cv::IMREAD_GRAYSCALE);
#ifdef PP_DEBUG
    //struct stat info;
    //if(fs::is_directory(img_save_path)) {
    //    if (fs::exists(img_save_path)) { // 判断文件夹是否存在
    //        std::cout << "Folder exists!" << std::endl;
    //    }
    //}
#endif // PP_DEBUG
}

CNumRec::CNumRec(const std::string template_dir = "./Template/") {
    // Load templates
    for (int i = 0; i < 10; i++) {
        cv::Mat template_img = cv::imread(template_dir + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);
        //Mat binary_img_i;
        //cv::threshold(template_img, binary_img_i, 250, 255, THRESH_BINARY_INV);
        m_template_list.push_back(template_img);
    }
    cv::String white_template_path = template_dir + "white_template1.bmp";
    m_img_white_gray = cv::imread(white_template_path, cv::IMREAD_GRAYSCALE);
#ifdef PP_DEBUG
    //struct stat info;
    //if (fs::is_directory(img_save_path)) {
    //    if (fs::exists(img_save_path)) { // 判断文件夹是否存在
    //        std::cout << "Folder exists!" << std::endl;
    //    }
    //}
#endif // PP_DEBUG
}

cv::Rect CNumRec::findWhiteArea(const cv::Mat& img_gray) {
    char buf[125];
    OutputDebugStringA("<<findWhiteArea>> enter findWhiteArea");
    cv::Mat img_gray_binary;
    cv::threshold(img_gray, img_gray_binary, 250, 255, cv::THRESH_BINARY);
#ifdef PP_DEBUG
    static int index;
    string file_name_debug = NumRecDebug + to_string(index) + ".bmp";
    cv::imwrite(file_name_debug, img_gray);
    map_img["findWhiteArea_img_gray_binary"] = img_gray_binary;
#endif // PP_DEBUG

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_gray_binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

#ifdef PP_DEBUG
    sprintf_s(buf, "<<findWhiteArea>> findContours ok");
    OutputDebugStringA(buf);
    cv::Mat img_bgr_contours;
    cv::cvtColor(img_gray, img_bgr_contours, cv::COLOR_GRAY2BGR);
    drawContours(img_bgr_contours, contours, -1, cv::Scalar(0, 0, 255), 2);
    map_img["findWhiteArea_contours"] = img_bgr_contours;
#endif // PP_DEBUG
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
        sprintf_s(buf, "<<findWhiteArea>> findWhiteArea got fiald");
        OutputDebugStringA(buf);
    }

    sprintf_s(buf, "<<findWhiteArea>> result_rect xywh %d %d %d %d", result_rect.x, result_rect.y, result_rect.width, result_rect.height);
    OutputDebugStringA(buf);
    sprintf_s(buf, "<<findWhiteArea>> findWhiteArea out");
    OutputDebugStringA(buf);
    return result_rect;
}

cv::Rect CNumRec::findNumArea(const cv::Mat& img_cut_binary_img) {
    char buf[125];
    OutputDebugStringA("<<findNumArea>> findNumArea  enter ");

    // Morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 3));
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 1));
    cv::Mat kernel7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 3));
    cv::Mat img_openning;
    cv::Mat img_closing;
    cv::Mat dilated_img;
    cv::morphologyEx(img_cut_binary_img, img_openning, cv::MORPH_OPEN, kernel3);
    cv::morphologyEx(img_openning, img_closing, cv::MORPH_CLOSE, kernel);
    cv::dilate(img_closing, dilated_img, kernel7, cv::Point(-1, -1), 1);

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(dilated_img, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect(contours_poly[i]);
    }

    // Find digit region
    cv::Rect result_rect(0, 1000, 0, 0);
    for (cv::Rect rect : boundRect)
    {
        if (rect.width / rect.height >= 4.0 && result_rect.y > rect.y && rect.y > 200)
        {
            result_rect = rect;
        }
    }
    sprintf_s(buf, "<<findNumArea>> findNumArea out");
    OutputDebugStringA(buf);
    return result_rect;
}

// x 投影
std::vector<int> CNumRec::x_projection(cv::Mat binary_img) {
    cv::Mat horizontal_projection;
    cv::reduce(binary_img, horizontal_projection, 1, cv::REDUCE_SUM, CV_32S);
    std::vector<int> result(2);
    for (int i = 0; i < horizontal_projection.rows; i++) {
        int cnt = horizontal_projection.at<int>(i, 0);
        if (cnt > 0) {
            result[0] = i;
            break;
        }
    }
    for (int i = horizontal_projection.rows - 1; i >= 0; i--) {
        int cnt = horizontal_projection.at<int>(i, 0);
        if (cnt > 0) {
            result[1] = i;
            break;
        }
    }
    return result;
}

// y 投影
vector<array<int, 2>> CNumRec::y_projection(cv::Mat binary_img_src) {
    sprintf_alg("<y_projection> y_projection enter");
    cv::Mat binary_img;
    cv::threshold(binary_img_src, binary_img, 250, 1, cv::THRESH_BINARY);
    cv::Mat vertical_projection;
    cv::reduce(binary_img, vertical_projection, 0, cv::REDUCE_SUM, CV_32S);
    int* arr = vertical_projection.ptr<int>(0);
    int arr_size = vertical_projection.cols;

    std::vector<std::array<int, 2>> white_area;
    int start = -1;
    for (int i = 0; i < arr_size; i++) // Iterate through horizontal projection array
    {
        int cnt = arr[i];
        if (cnt > 1) {
            if (start == -1) {
                start = i;
            }
        }
        else if (cnt == 0 && start != -1) {
            white_area.push_back({ start, i - 1 });
            start = -1;
        }
    }
    std::vector<std::array<int, 2>> white_area_merge;
    for (int i = 0; i < white_area.size(); i++) {
        auto area = white_area[i];
        if (area[1] - area[0] > 12) {
            white_area_merge.push_back(area);
        }
        else {
            auto area_pre = white_area_merge.back();
            if (area[0] - area_pre[1] <= 6 && area[1] - area_pre[0] < 30) {
                white_area_merge.back()[1] = area[1];
            }
            else {
                white_area_merge.push_back(area);
            }
        }
    }
    sprintf_alg("<y_projection> y_projection out");
    return white_area_merge;
}

vector<cv::Rect> CNumRec::getNumBoxByProjection(cv::Mat binary_img) {
    sprintf_alg("<<<<<<<<getNumBoxByProjection>>>>>>>>>>> getNumBoxByProjection enter");
    std::vector<std::array<int, 2>> num_area_x_list = y_projection(binary_img);
    std::vector<std::vector<int>> num_area_y_list;
    vector<cv::Rect> resultRect;
    for (auto num_area_x : num_area_x_list) {
        cv::Mat binary_img_oneNum = binary_img(cv::Rect(num_area_x[0], 0, num_area_x[1] - num_area_x[0], binary_img.rows));
        std::vector<int> num_area_y = x_projection(binary_img_oneNum);
        num_area_y_list.push_back({ num_area_y[0], num_area_y[1] });
        resultRect.push_back(cv::Rect(num_area_x[0], num_area_y[0], num_area_x[1] - num_area_x[0], num_area_y[1] - num_area_y[0]));
        sprintf_alg("NumBox: xywh: %d %d %d %d", num_area_x[0], num_area_y[0], num_area_x[1] - num_area_x[0], num_area_y[1] - num_area_y[0]);
    }
    sprintf_alg("<<<<<<<<getNumBoxByProjection>>>>>>>>>>> getNumBoxByProjection out");
    return resultRect;
}
vector<cv::Rect> CNumRec::getNumBoxByContours(cv::Mat binary_img) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Rect> boundRect(contours.size());
    for (int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours[i], 3, true);
        boundRect[i] = cv::boundingRect(contours[i]);
    }
    std::sort(boundRect.begin(), boundRect.end(), [](cv::Rect a, cv::Rect b) { return a.x < b.x; });
    //cv::Rect* result = new cv::Rect[boundRect.size()];
    //std::copy(boundRect.begin(), boundRect.end(), result);
    return boundRect;
}

bool CNumRec::processImage(CNumRecPara& num_rec_para) {
    return true;
}

bool CNumRec::processImage(const cv::Mat& img_bgr, string& str_result) {
    char buf[125];
    sprintf_alg("打印 selfname");
    OutputDebugStringA("<<<<<<<<processImage>>>>>>>>>>> processImage enter >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    reset();
#ifdef PP_DEBUG
    //Create a time_t object and get the current time
    time_t now = time(0);
    //Create a tm struct to hold the current time
    tm ltm;
    localtime_s(&ltm, &now);
    std::stringstream ss;
    ss << std::put_time(&ltm, "%Y%m%d%H%M");
    std::string str_time = ss.str();
#endif // PP_DEBUG

    if (img_bgr.empty()) {
        return false;
    }
    cv::Mat img_gray;
    if (img_bgr.channels() == 1) // check if image is grayscale
    {
        img_gray = img_bgr; // assign the grayscale image to gray_img
    }
    else // if image is not grayscale
    {
        cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY); // convert color image to grayscale
    }
    // 查找白色区域
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);
    cv::Mat img_cut = img_gray(whiteArea_rect);
    //resize(img_cut, img_cut, cv::Size(693, 417), 0, 0, cv::INTER_LINEAR);
    cv::Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 250, 255, cv::THRESH_BINARY_INV);

    // 查找数字区域
    cv::Rect result_rect = findNumArea(img_cut_binary_img);
    cv::Mat num_img = img_cut(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));
    cv::Mat binary_img;
    cv::threshold(num_img, binary_img, 190, 255, cv::THRESH_BINARY_INV);

    // 大小标准化
    //int num_img_h = binary_img.rows;
    //int num_img_w = binary_img.cols;
    //float ratio = num_img_h / 35.0f;
    //int num_img_w_new = static_cast<int>(num_img_w / ratio);
    //cv::resize(binary_img, binary_img, cv::Size(num_img_w_new, 35), 0,0,cv::INTER_LINEAR);
    //cv::resize(num_img, num_img, cv::Size(num_img_w_new, 35), 0, 0, cv::INTER_LINEAR);

    vector<cv::Rect> result_boxes = getNumBoxByProjection(binary_img);
    // 判断模版存在
    if (m_template_list[0].empty()) {
        OutputDebugStringA("<<<<<<processImage>>>>>>  template file is not exist!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        return false;
    }
    for (int index = 0; index < 7; index++) {
        cv::Rect box = result_boxes[index];
        int x = box.x, y = box.y, w = box.width, h = box.height;
        int y_new = y - 1;
        int h_new = h + 2;
        if (y_new < 0) {
            y_new = 0;
        }
        if (y_new + h_new > num_img.rows) {
            h_new = num_img.rows - y_new;
        }
        int x_new = x - 2;
        if (x_new < 0) {
            x_new = 0;
        }
        int w_new = w + 4;
        cv::Mat img_tmp = num_img(cv::Rect(x_new, y_new, w_new, h_new));
        cv::resize(img_tmp, img_tmp, m_template_list[0].size(), 0, 0, cv::INTER_LINEAR);

#ifdef PP_DEBUG
        string img_num_little = NumRecDebug + str_time + "_img_num_little_" + to_string(index) + ".bmp";
        cv::imwrite(img_num_little, img_tmp);
#endif // PP_DEBUG

        double max_score = 0;
        int num_result = -1;
        std::vector<double> scores;
        for (int i = 0; i < m_template_list.size(); i++) {
            cv::Mat template_img = m_template_list[i];
            cv::Mat res;
            cv::matchTemplate(img_tmp, template_img, res, cv::TM_CCOEFF_NORMED);
            double max_val;
            cv::minMaxLoc(res, nullptr, &max_val);
            scores.push_back(max_val);
            if (max_val > max_score) {
                max_score = max_val;
                num_result = i;
            }
        }

        std::vector<int> scores_indices(scores.size());
        std::iota(scores_indices.begin(), scores_indices.end(), 0);
        std::sort(scores_indices.begin(), scores_indices.end(), [&](int i, int j) { return scores[i] < scores[j]; });
        sprintf_s(buf, "scores_indices -2 -1 : %f %f", scores[scores_indices[scores_indices.size() - 2]], scores[scores_indices[scores_indices.size() - 1]]);
        OutputDebugStringA(buf);
        if (w > 12) {

            cv::Mat img_tmp_binary = binary_img(cv::Rect(x_new, y_new, w_new, h_new));
            num_result = scores_indices[scores_indices.size() - 1];
            if (num_result == 1) {
                num_result = scores_indices[scores_indices.size() - 2];
            }
            //if (num_result == 6) {
            //    cv::Mat img_center = img_tmp_binary(cv::Rect(8, 10, 5, 12));
            //    cv::imwrite("img0.bmp", img_center);
            //    int sum = cv::sum(img_center)[0];
            //    sprintf_s(buf, "0000000   sum : %d", sum);
            //}
            if (num_result == 0) {
                cv::Mat img_center = img_tmp_binary(cv::Rect(8, 10, 5, 12));
                double sum = cv::sum(img_center)[0];
                sprintf_s(buf, "0000000   sum : %.3f", sum);
                OutputDebugStringA(buf);
                if (sum > 1000) {
                    if (scores_indices[scores_indices.size() - 2] == 9) {
                        num_result = 9;
                    }
                    if (scores_indices[scores_indices.size() - 2] == 6) {
                        num_result = 6;
                    }
                    if (scores_indices[scores_indices.size() - 2] == 8) {
                        num_result = 8;
                    }
                }

            }
        }
        else {
            num_result = 1;
        }
        std::cout << index << " scores: ";
        for (double score : scores) {
            std::cout << score << " ";
        }
        std::cout << std::endl;
        str_result = str_result + std::to_string(num_result);
    }

#ifdef PP_DEBUG
    static int i = 0;
    string file_path = NumRecBGR;
    file_path += str_time + "_" + to_string(i) + "_" + str_result;
    string save_path = file_path + ".bmp";
    cv::imwrite(save_path, img_bgr);
    cv::imwrite("img.bmp", img_bgr);
    // 保存debug图
    map_img["processImage_NumArea_binary_img"] = binary_img;
    file_path = NumRecDebug;
    file_path += str_time + "_" + to_string(i) + "_" + str_result;
    for (const auto& pair : map_img) {
        string key = pair.first;
        cv::Mat img = pair.second;
        save_path = file_path + "_" + key + ".bmp";
        cv::imwrite(save_path, img);
    }
#endif // PP_DEBUG 

    sprintf_s(buf, "<<<<<<processImage>>>>>> str_result=%s", str_result.c_str());
    OutputDebugStringA(buf);
    OutputDebugStringA("<<<<<<<<processImage>>>>>>>>>>> processImage out >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>");
    return true;
}

string CNumRec::processImage(const cv::Mat& img_bgr) {
    std::string str_result = "";
    processImage(img_bgr, str_result);
    return str_result;
}
string CNumRec::processImage1(const cv::Mat& img_gray) {
    //double startTime = clock();//计时开始 
    // 查找白色区域
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);
    cv::Mat img_cut = img_gray(whiteArea_rect);
    resize(img_cut, img_cut, cv::Size(693, 417), 0, 0, cv::INTER_LINEAR);
    cv::Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 200, 255, cv::THRESH_BINARY_INV);
    //cv::imshow("img_cut_binary_img", img_cut_binary_img);
    //moveWindow("img_cut_binary_img", 100, 100);
    //cv::waitKey(0);

    cv::Rect result_rect = findNumArea(img_cut_binary_img);
    cv::Mat binary_img = img_cut_binary_img(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));
    cv::Mat num_img = img_cut(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));

    //cv::imshow("Binary image", binary_img);
    //moveWindow("Binary image", 100, 100);
    //cv::waitKey(0);
    // Apply vertical projection to the binary image to get the sum of white pixels in each column
    cv::threshold(binary_img, binary_img, 250, 1, cv::THRESH_BINARY);

    cv::Mat vertical_projection;
    reduce(binary_img, vertical_projection, 0, cv::REDUCE_SUM, CV_32S);

    int* arr = vertical_projection.ptr<int>(0);
    int arr_size = vertical_projection.cols;

    // Now you can access the elements of the array using the [] operator
    vector<int> white_x_min;
    vector<int> white_x_max;
    int step = 10;
    bool run_one_flg = false;
    for (int i = 0; i < arr_size; i++)
    {
        int element = arr[i];
        // Do something with the element
        if (element == 0) {
            if (run_one_flg && step > 2) {
                white_x_max.push_back(i);
                step = 0;
                run_one_flg = false;
            }
            else {
                step++;
            }
        }
        else if (element > 0) {
            if (!run_one_flg && step > 2) {
                white_x_min.push_back(i);
                step = 0;
                run_one_flg = true;
            }
            else {
                step++;
            }
        }
    }

    // Recognize digits
    std::vector<std::vector<int>> result;     // class, xmin, ymin
    string str_result = "";
    int xmin_ = 0;
    int xmax_ = 0;
    for (int index = 0; index < 7; index++)
    {
        xmin_ = white_x_min[index] - 3;
        if (xmin_ < 0) {
            xmin_ = 0;
        }

        xmax_ = white_x_max[index];
        cv::Mat img_tmp = num_img(cv::Rect(xmin_, 0, xmax_ - xmin_ + 6, num_img.rows));
        double max_score = 0;
        cv::Mat loc;
        int xmin__ = 0;
        std::vector<int> res_loc;
        for (int i = 0; i < m_template_list.size(); i++)
        {
            cv::Mat template_img = m_template_list[i];
            cv::Mat res;
            cv::matchTemplate(img_tmp, template_img, res, cv::TM_CCOEFF_NORMED);
            double max_;
            cv::minMaxLoc(res, NULL, &max_);
            if (max_ > max_score)
            {
                double threshold = max_;
                cv::findNonZero(res >= threshold, loc);
                xmin__ = loc.at<cv::Point>(0).x;
                int ymin_ = loc.at<cv::Point>(0).y;
                max_score = max_;
                int max_index = i;
                res_loc = { i, xmin_ + xmin__, ymin_ };
            }
        }
        str_result = str_result + to_string(res_loc[0]);
        result.push_back(res_loc);
    }

    for (const auto& vec : result) {
        for (const auto& elem : vec) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    // std::cout << " time: " << clock() - startTime << std::endl;
    // namedWindow("Binary image", WINDOW_NORMAL);
    cv::Mat grayscale_image;
    cv::cvtColor(num_img, grayscale_image, cv::COLOR_GRAY2BGR);
    for (int index = 0; index < 7; index++) {
        xmin_ = white_x_min[index];
        xmax_ = white_x_max[index];
        cv::line(grayscale_image, cv::Point(xmin_, 0), cv::Point(xmin_, num_img.rows), cv::Scalar(0, 0, 255), 1);
        cv::line(grayscale_image, cv::Point(xmax_, 0), cv::Point(xmax_, num_img.rows), cv::Scalar(0, 0, 255), 1);
    }
    cv::imshow("Binary image", grayscale_image);
    cv::moveWindow("Binary image", 100, 100);
    cv::waitKey(0);
    return str_result;
}

/// <summary>
/// 测试代码
/// </summary>
void NumRecTest() {
    cv::String dir_root = "D:/02dataset/01work/05nanjingLG/03NumRec/";
    //cv::String dir_root = "D:/02dataset/01work/01TuoPanLJ/03NumRec/";
    cv::String dir_template = dir_root + "/template/";
    cv::String dir_imgall = dir_root + "/imgall0422/";
    cv::String img_path = dir_imgall + "img_3_0538062.bmp";
    cv::Mat img_gray = cv::imread(img_path, cv::IMREAD_GRAYSCALE);
    CNumRec nr = CNumRec(dir_template);
    string str_result;

    double startTime = clock();//计时开始
    bool result = nr.processImage(img_gray, str_result);
    cout << " time: " << clock() - startTime << endl;
    cout << "result " << result << endl;
    cout << "str_result " << str_result << endl;
}
