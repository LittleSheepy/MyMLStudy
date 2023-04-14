// NumRec.cpp
#include "NumRec.h"
#include <iostream>




CNumRec::CNumRec(const std::string template_dir) {
    // Load templates
    for (int i = 0; i < 10; i++) {
        cv::Mat template_img = cv::imread(template_dir + std::to_string(i) + ".bmp", cv::IMREAD_GRAYSCALE);
        Mat binary_img_i;
        cv::threshold(template_img, binary_img_i, 250, 255, THRESH_BINARY_INV);
        template_list.push_back(binary_img_i);
    }
    String white_template_path = template_dir + "white_template1.bmp";
    img_white_gray = cv::imread(white_template_path, IMREAD_GRAYSCALE);
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
void CNumRec::processImage(const cv::Mat& img_gray) {
    double startTime_total = clock();//计时开始
    double startTime = clock();//计时开始
    cv::Rect whiteArea_rect;
    whiteArea_rect = findWhiteArea(img_gray);
    //// Image preprocessing
    //double threshold = 0.8;
    //Mat img_gray_ = img_gray(Rect(1000, 900, 1200, 600));
    //cv::Mat res;
    //cv::matchTemplate(img_gray_, img_white_gray, res, cv::TM_CCOEFF_NORMED);
    //double max_;
    //cv::minMaxLoc(res, NULL, &max_);
    //threshold = max_;
    //cv::Mat loc;
    //cv::findNonZero(res >= threshold, loc);
    //int xmin = loc.at<cv::Point>(0).x + 1000;
    //int ymin = loc.at<cv::Point>(0).y + 900;
    //int h = img_white_gray.rows;
    //int w = img_white_gray.cols;

    cv::Mat img_cut = img_gray(whiteArea_rect);

    std::cout << "img_cut time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始

    cv::Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 250, 255, cv::THRESH_BINARY_INV);

    // Morphological operations
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(30, 3));
    cv::Mat kernel3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::Mat img_closing;
    cv::morphologyEx(img_cut_binary_img, img_closing, cv::MORPH_CLOSE, kernel);
    cv::Mat dilated_img;
    cv::dilate(img_closing, dilated_img, kernel3, cv::Point(-1, -1), 1);

    std::cout << "dilate time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始

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

    cv::Mat binary_img = img_cut_binary_img(cv::Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));

    std::cout << "binary_img time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始

    // Recognize digits
    std::vector<std::vector<int>> result;     // class, xmin, ymin
    int xmin_ = 0;
    while (true)
    {
        cv::Mat img_tmp = binary_img(cv::Rect(xmin_, 0, 30, binary_img.rows));
        double max_score = 0;
        cv::Mat loc;
        int xmin__ = 0;
        std::vector<int> res_loc;
        for (int i = 0; i < template_list.size(); i++)
        {
            cv::Mat template_img = template_list[i];
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
        xmin_ = xmin_ + xmin__ + 25;
        if (xmin_ + 40 > binary_img.cols || result.size() >= 7)
        {
            break;
        }
    }
    for (const auto& vec : result) {
        for (const auto& elem : vec) {
            std::cout << elem << " ";
        }
        std::cout << std::endl;
    }
    std::cout << " time: " << clock() - startTime << std::endl;
    std::cout << " total time: " << clock() - startTime_total << std::endl;
    //namedWindow("Binary image", WINDOW_NORMAL);
    cv::imshow("Binary image", binary_img);
    moveWindow("Binary image", 100, 100);
    cv::waitKey(0);
}