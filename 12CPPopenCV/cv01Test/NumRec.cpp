// NumRec.cpp
#include "NumRec.h"
#include <iostream>
#include <numeric>



CNumRec::CNumRec(const std::string template_dir) {
    // Load templates
    for (int i = 0; i < 10; i++) {
        cv::Mat template_img = cv::imread(template_dir + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);
        //Mat binary_img_i;
        //cv::threshold(template_img, binary_img_i, 250, 255, THRESH_BINARY_INV);
        m_template_list.push_back(template_img);
    }
    String white_template_path = template_dir + "white_template1.bmp";
    m_img_white_gray = cv::imread(white_template_path, IMREAD_GRAYSCALE);
}

Rect CNumRec::findWhiteArea(const cv::Mat& img_gray) {
    cv::Mat img_gray_binary;
    cv::threshold(img_gray, img_gray_binary, 250, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(img_gray_binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<cv::Point>> contours_poly(contours.size());
    std::vector<cv::Rect> boundRect(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = cv::boundingRect(contours_poly[i]);
    }
    // Filter out contours with an area less than 100
    vector<Rect> filteredRects;
    for (Rect rect : boundRect)
    {
        if (rect.area() >= 150000 && rect.area() < 350000)
        {
            filteredRects.push_back(rect);
        }
    }
    //// Find digit region
    //cv::Rect result_rect(0, 1000, 0, 0);
    //for (cv::Rect rect : filteredRects)
    //{
    //    if (rect.width * rect.height >= 5.0 && result_rect.y > rect.y && rect.y > 200)
    //    {
    //        result_rect = rect;
    //    }
    //}
    if (filteredRects.size() > 0) {
        return filteredRects[0];
    }
    else {
        return Rect(0, 0, 0, 0);
    }
}

Rect CNumRec::findNumArea(const cv::Mat& img_cut_binary_img) {


    // Morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 3));
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat img_closing;
    cv::morphologyEx(img_cut_binary_img, img_closing, cv::MORPH_CLOSE, kernel);
    cv::Mat dilated_img;
    cv::dilate(img_closing, dilated_img, kernel3, cv::Point(-1, -1), 1);

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
    return white_area_merge;
}

vector<Rect> CNumRec::getNumBoxByProjection(cv::Mat binary_img) {
    std::vector<std::array<int, 2>> num_area_x_list = y_projection(binary_img);
    std::vector<std::vector<int>> num_area_y_list;
    vector<Rect> resultRect;
    for (auto num_area_x : num_area_x_list) {
        cv::Mat binary_img_oneNum = binary_img(cv::Rect(num_area_x[0], 0, num_area_x[1] - num_area_x[0], binary_img.rows));
        std::vector<int> num_area_y = x_projection(binary_img_oneNum);
        num_area_y_list.push_back({ num_area_y[0], num_area_y[1] });
        resultRect.push_back(Rect(num_area_x[0], num_area_y[0], num_area_x[1] - num_area_x[0], num_area_y[1] - num_area_y[0]));
    }
    return resultRect;
}
vector<Rect> CNumRec::getNumBoxByContours(cv::Mat binary_img) {
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
string CNumRec::processImage(const cv::Mat& img_gray) {
    // 查找白色区域
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);
    cv::Mat img_cut = img_gray(whiteArea_rect);
    resize(img_cut, img_cut, Size(693, 417), 0, 0, INTER_LINEAR);
    cv::Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 200, 255, cv::THRESH_BINARY_INV);

    // 查找数字区域
    cv::Rect result_rect = findNumArea(img_cut_binary_img);
    cv::Mat num_img = img_cut(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));
    cv::Mat binary_img;
    cv::threshold(num_img, binary_img, 190, 255, cv::THRESH_BINARY_INV);

    // 大小标准化
    int num_img_h = binary_img.rows;
    int num_img_w = binary_img.cols;
    float ratio = num_img_h / 35.0f;
    int num_img_w_new = static_cast<int>(num_img_w / ratio);
    cv::resize(binary_img, binary_img, cv::Size(num_img_w_new, 35));
    cv::resize(num_img, num_img, cv::Size(num_img_w_new, 35));

    vector<Rect> result_boxes = getNumBoxByProjection(binary_img);

    std::string str_result = "";
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
        if (w > 12) {
            num_result = scores_indices[scores_indices.size() - 1];
            if (num_result == 1) {
                num_result = scores_indices[scores_indices.size() - 2];
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
        str_result += std::to_string(num_result);
    }
    return str_result;
}
string CNumRec::processImage1(const cv::Mat & img_gray) {
    //double startTime = clock();//计时开始
    // 查找白色区域
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);
    cv::Mat img_cut = img_gray(whiteArea_rect);
    resize(img_cut, img_cut, Size(693, 417), 0, 0, INTER_LINEAR);
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

    Mat vertical_projection;
    reduce(binary_img, vertical_projection, 0, REDUCE_SUM, CV_32S);

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
    string str_result="";
    int xmin_ = 0;
    int xmax_ = 0;
    for (int index = 0; index < 7; index++)
    {
        xmin_ = white_x_min[index]-3;
        if (xmin_ < 0) {
            xmin_ = 0;
        }

        xmax_ = white_x_max[index];
        cv::Mat img_tmp = num_img(cv::Rect(xmin_, 0, xmax_ - xmin_+6, num_img.rows));
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
    Mat grayscale_image;
    cv::cvtColor(num_img, grayscale_image, COLOR_GRAY2BGR);
    for (int index = 0; index < 7; index++) {
        xmin_ = white_x_min[index];
        xmax_ = white_x_max[index];
        line(grayscale_image, Point(xmin_, 0), Point(xmin_, num_img.rows), Scalar(0, 0, 255), 1);
        line(grayscale_image, Point(xmax_, 0), Point(xmax_, num_img.rows), Scalar(0, 0, 255), 1);
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
    String dir_root = "D:/02dataset/01work/05nanjingLG/03NumRec/";
    String dir_template = dir_root + "/template/";
    String dir_imgall = dir_root + "/imgall/";
    String img_path = dir_imgall + "Image_20230415092911313.bmp";
    String white_template_path = dir_root + "Image_20230415092911313.bmp";
    Mat img_gray = imread(img_path, cv::IMREAD_GRAYSCALE);
    CNumRec nr = CNumRec(dir_template);
    string str_result;

    double startTime = clock();//计时开始
    str_result = nr.processImage(img_gray);
    cout << " time: " << clock() - startTime << endl;
    cout << str_result << endl;
}
