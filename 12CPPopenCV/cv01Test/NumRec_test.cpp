#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include "Numrec.h"
using namespace cv;
using namespace std;


void NumRec()
{
    std::clock_t startTime = clock();//计时开始
    std::clock_t startTime_total = clock();//计时开始
    String dir_root = "D:/03GitHub/00myGitHub/MyMLStudy/ml00project/pj2LG/numRec/";
    String img_path = dir_root + "black_0074690_CM3_1.bmp";
    String white_template_path = dir_root + "white_template3.bmp";
    // 模板
    vector<Mat> template_list;
    for (int i = 0; i < 10; i++)
    {
        String img_i_path = dir_root + to_string(i) + ".bmp";
        Mat img_i = imread(img_i_path, IMREAD_GRAYSCALE);
        Mat binary_img_i;
        cv::threshold(img_i, binary_img_i, 250, 255, THRESH_BINARY_INV);
        template_list.push_back(binary_img_i);
    }

    std::cout << "read TMP time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    // 文字位置
    vector<Point> nums_pos;
    Mat img_gray = imread(img_path, IMREAD_GRAYSCALE);
    Mat img_white_gray = imread(white_template_path, IMREAD_GRAYSCALE);

    Mat img_gray_ = img_gray(Rect(1000,900, 1200, 600));
    Mat res;
    matchTemplate(img_gray_, img_white_gray, res, TM_CCOEFF_NORMED);
    double max_;
    minMaxLoc(res, NULL, &max_);
    // 画出匹配结果
    double threshold = max_;
    Mat loc;
    findNonZero(res >= threshold, loc);
    int xmin = loc.at<Point>(0).x+1000;
    int ymin = loc.at<Point>(0).y+900;
    int h = img_white_gray.rows;
    int w = img_white_gray.cols;
    Mat img_cut = img_gray(Rect(xmin, ymin, w, h));

    std::cout << "img_cut time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    Mat img_cut_binary_img;
    cv::threshold(img_cut, img_cut_binary_img, 250, 255, THRESH_BINARY_INV);
    // 定义水平膨胀核
    Mat kernel = getStructuringElement(MORPH_RECT, Size(30, 3));
    Mat kernel3 = getStructuringElement(MORPH_RECT, Size(7, 7));
    // 水平膨胀
    Mat img_closing;
    morphologyEx(img_cut_binary_img, img_closing, MORPH_CLOSE, kernel);
    Mat dilated_img;
    dilate(img_closing, dilated_img, kernel3, Point(-1, -1), 1);

    std::cout << "dilate time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    // 找轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dilated_img, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);

    vector<vector<Point>> contours_poly(contours.size());
    vector<Rect> boundRect(contours.size());
    for (size_t i = 0; i < contours.size(); i++)
    {
        approxPolyDP(contours[i], contours_poly[i], 3, true);
        boundRect[i] = boundingRect(contours_poly[i]);
    }

    // 按照特征找
    Rect result_rect(0, 1000, 0, 0);
    for (Rect rect : boundRect)
    {
        if (rect.width / rect.height >= 5.0 && result_rect.y > rect.y && rect.y > 200)
        {
            result_rect = rect;
        }
    }

    Mat binary_img = img_cut_binary_img(Rect(result_rect.x, result_rect.y, result_rect.width, result_rect.height));

    std::cout << "binary_img time: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始

    vector<vector<int>> result;     // class, xmin, ymin
    int xmin_ = 0;
    while (true)
    {
        Mat img_tmp = binary_img(Rect(xmin_, 0, 40, binary_img.rows));
        double max_score = 0;
        Mat loc;
        int xmin__ = 0;
        vector<int> res_loc;
        for (int i = 0; i < template_list.size(); i++)
        {
            Mat template_img = template_list[i];
            Mat res;
            matchTemplate(img_tmp, template_img, res, TM_CCOEFF_NORMED);
            double max_;
            minMaxLoc(res, NULL, &max_);
            if (max_ > max_score)
            {
                threshold = max_;
                findNonZero(res >= threshold, loc);
                xmin__ = loc.at<Point>(0).x;
                int ymin_ = loc.at<Point>(0).y;
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
    imshow("Binary image", binary_img);
    waitKey(0);
}
void NumRecTest() {
    String dir_root = "F:/sheepy/00MyMLStudy/ml00project/pj2LG/numRec/";
    String img_path = dir_root + "black_0074690_CM3_1.bmp";
    String white_template_path = dir_root + "white_template3.bmp";
    Mat img_gray = imread(img_path, cv::IMREAD_GRAYSCALE);
    CNumRec nr = CNumRec(dir_root);
    nr.processImage(img_gray);
}

