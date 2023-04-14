// NumRec.cpp
#include "NumRec.h"
#include <iostream>


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
        if (rect.area() >= 240000 && rect.area() < 280000)
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

    //std::cout << "dilate time: " << clock() - startTime << std::endl;
    //startTime = clock();//计时开始

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
        if (rect.width / rect.height >= 5.0 && result_rect.y > rect.y && rect.y > 200)
        {
            result_rect = rect;
        }
    }
    return result_rect;
}

void CNumRec::processImage(const cv::Mat& img_gray) {
    //double startTime = clock();//计时开始
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);

    cv::Mat img_cut = img_gray(whiteArea_rect);
    cv::Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 250, 255, cv::THRESH_BINARY_INV);

    cv::Rect result_rect = findNumArea(img_cut_binary_img);
    cv::Mat binary_img = img_cut_binary_img(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));
    cv::Mat num_img = img_cut(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));

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
            if (run_one_flg && step > 5) {
                white_x_max.push_back(i);
                step = 0;
                run_one_flg = false;
            }
            else {
                step++;
            }
        }
        else if (element > 0) {
            if (!run_one_flg && step > 5) {
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
    int xmin_ = 0;
    int xmax_ = 0;
    for (int index = 0; index < 7; index++)
    {
        xmin_ = white_x_min[index]-3;
        if (xmin_ < 0) {
            xmin_ = 0;
        }

        xmax_ = white_x_max[index];
        cv::Mat img_tmp = num_img(cv::Rect(xmin_, 0, xmax_ - xmin_+3, num_img.rows));
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
        result.push_back(res_loc);
    }
    for (const auto& vec : result) {
        for (const auto& elem : vec) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    //std::cout << " time: " << clock() - startTime << std::endl;
    ////namedWindow("Binary image", WINDOW_NORMAL);
    //Mat grayscale_image;
    //cvtColor(num_img, grayscale_image, COLOR_GRAY2BGR);
    //for (int index = 0; index < 7; index++) {
    //    xmin_ = white_x_min[index];
    //    xmax_ = white_x_max[index];
    //    line(grayscale_image, Point(xmin_, 0), Point(xmin_, num_img.rows), Scalar(0, 0, 255), 1);
    //    line(grayscale_image, Point(xmax_, 0), Point(xmax_, num_img.rows), Scalar(0, 0, 255), 1);
    //}
    //cv::imshow("Binary image", grayscale_image);
    //moveWindow("Binary image", 100, 100);
    //cv::waitKey(0);
}

/// <summary>
/// 测试代码
/// </summary>

//void NumRecTest() {
//    String dir_root = "F:/sheepy/00MyMLStudy/ml00project/pj2LG/numRec/";
//    String img_path = dir_root + "black_0074690_CM3_1.bmp";
//    String white_template_path = dir_root + "white_template3.bmp";
//    Mat img_gray = imread(img_path, cv::IMREAD_GRAYSCALE);
//    CNumRec nr = CNumRec(dir_root);
//    nr.processImage(img_gray);
//}
