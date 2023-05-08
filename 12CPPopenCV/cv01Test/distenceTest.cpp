#include <opencv2/opencv.hpp>
#include <iostream>
#include<ctime>

#include "VectorialAngle.hpp"
using namespace std;
using namespace cv;


map<string, Point> getColumnPoint(Mat img_gray, int point_x, int point_y_start = 0) {
    int h = img_gray.rows;
    int y = point_y_start;
    map<string, Point> ColumnPoints;
    for (int point_y = 0; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) > 200) {
            y = point_y;
            ColumnPoints["whitetop"] = Point(point_x, point_y);
            break;
        }
    }
    for (int point_y = y + 1; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) < 160) {
            y = point_y;
            ColumnPoints["blacktop"] = Point(point_x, point_y);
            break;
        }
    }
    for (int point_y = y + 1; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) > 200) {
            y = point_y;
            ColumnPoints["blackbottom"] = Point(point_x, point_y);
            break;
        }
    }
    for (int point_y = y + 1; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) < 100) {
            y = point_y;
            ColumnPoints["whitebottom"] = Point(point_x, point_y);
            break;
        }
    }
    return ColumnPoints;
}

map<string, Point> getEarPoint(Mat img_gray, int point_x) {
    int h = img_gray.rows;
    int y = 0;
    map<string, Point> EarPoints;
    for (int point_y = 0; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) > 200) {
            y = point_y;
            EarPoints["top"] = Point(point_x, point_y);
            break;
        }
    }
    for (int point_y = y + 1; point_y < h; point_y++) {
        if (img_gray.at<uchar>(point_y, point_x) < 50) {
            y = point_y;
            EarPoints["bottom"] = Point(point_x, point_y);
            break;
        }
    }
    return EarPoints;
}

map<string, Point> getRowWhitePoint(Mat img_gray, int point_y) {
    int w = img_gray.cols;
    map<string, Point> RowPoints;
    for (int point_x = 0; point_x < w; point_x++) {
        if (img_gray.at<uchar>(point_y, point_x) > 200) {
            RowPoints["left"] = Point(point_x, point_y);
            break;
        }
    }
    for (int point_x = w - 1; point_x > 0; point_x--) {
        if (img_gray.at<uchar>(point_y, point_x) > 200) {
            RowPoints["right"] = Point(point_x, point_y);
            break;
        }
    }
    return RowPoints;
}

void line_angle(Point& p1, Point& p2, Point& q1, Point& q2, double* angle, Point* point) {
    // 计算两条直线的斜率
    float k1 = (p2.y - p1.y) / (p2.x - p1.x + 0.00001);
    float k2 = (q2.y - q1.y) / (q2.x - q1.x + 0.00001);
    // 计算两条直线的夹角
    *angle = atan((k2 - k1) / (1 + k1 * k2)) * 180 / CV_PI;
    // 计算两条直线的交点
    point->x = (k1 * p1.x - k2 * q1.x + q1.y - p1.y) / (k1 - k2);
    //point->y = k1 * (point->x - p1.x) + p1.y;
    point->y = k2 * (point->x - q2.x) + q2.y;
}

void distenceTest() {
    std::clock_t startTime = clock();//计时开始
    std::clock_t startTime2 = clock();//计时开始
    string img_path = "D:/04DataSets/04/box1.jpg";
    Mat img_bgr = imread(img_path);
    Mat img_gray = imread(img_path, IMREAD_GRAYSCALE);
    cv::imshow("img_gray", img_gray);
    std::cout << "img_read: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    // 计算 关键点
    map<string, Point> ColumnKeyPoints = getColumnPoint(img_gray, 220);
    int RowPoints_y = ColumnKeyPoints["whitetop"].y
        + int((ColumnKeyPoints["blacktop"].y - ColumnKeyPoints["whitetop"].y) / 2);
    map<string, Point> RowKeyPoints = getRowWhitePoint(img_gray, RowPoints_y);

    // Column 获取点列表
    vector<map<string, Point>> ColumnPointsList;
    int point_y_start = ColumnKeyPoints["whitetop"].y - 20;
    int point_x_start = RowKeyPoints["left"].x + 5;
    int point_x_end = RowKeyPoints["right"].x - 5;
    for (int i = point_x_start; i <= point_x_end; i += 5) {
        int ColumnPointsTmp_x = i;
        map<string, Point> ColumnPointsTmp = getColumnPoint(img_gray, ColumnPointsTmp_x, point_y_start);
        ColumnPointsList.push_back(ColumnPointsTmp);
    }
    // Column 拟合直线
    vector<Point> ColWhiteTopPointsList;
    for (auto ColumnPoints : ColumnPointsList) {

        ColWhiteTopPointsList.push_back(ColumnPoints["whitetop"]);

    }
    Vec4f LineWhiteTop;
    fitLine(ColWhiteTopPointsList, LineWhiteTop, cv::DIST_L2, 0, 0.01, 0.01);

    // 计算Row 关键点
    int RowPointsOne_y = ColumnKeyPoints["whitetop"].y
        + int((ColumnKeyPoints["blacktop"].y - ColumnKeyPoints["whitetop"].y) / 2);
    map<string, Point> RowPointsOne = getRowWhitePoint(img_gray, RowPointsOne_y);

    int RowPointsTwo_y = ColumnKeyPoints["blackbottom"].y
        + int((ColumnKeyPoints["whitebottom"].y - ColumnKeyPoints["blackbottom"].y) / 2);
    map<string, Point> RowPointsTwo = getRowWhitePoint(img_gray, RowPointsTwo_y);

    int EarPointsOne_x = RowPointsOne["right"].x + 10;
    map<string, Point> EarPointsOne = getEarPoint(img_gray, EarPointsOne_x);
    int EarPointsTwo_x = RowPointsOne["right"].x + 60;
    map<string, Point> EarPointsTwo = getEarPoint(img_gray, EarPointsTwo_x);

    std::cout << "Column&Row: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    // 耳朵
    vector<map<string, Point>> EarPointsList;
    for (int i = 10; i <= 61; i += 5) {
        int EarPointsTmp_x = RowPointsOne["right"].x + i;
        map<string, Point> EarPointsTmp = getEarPoint(img_gray, EarPointsTmp_x);
        EarPointsList.push_back(EarPointsTmp);
    }

    std::cout << "Ear: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    // ear 拟合直线
    vector<Point> earTopPointsList;
    for (auto EarPoints : EarPointsList) {

        earTopPointsList.push_back(EarPoints["top"]);
    }
    //Mat earTopPointsMat = Mat(earTopPointsList);
    Vec4f output;
    fitLine(earTopPointsList, output, cv::DIST_L2, 0, 0.01, 0.01);

    std::cout << "fitLine: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始
    double angle;
    Point point;
    line_angle(RowPointsOne["right"], RowPointsTwo["right"],
        EarPointsOne["top"], EarPointsTwo["top"],
        &angle, &point);

    std::cout << "line_angle: " << clock() - startTime << std::endl;
    startTime = clock();//计时开始


    std::cout << "end: " << clock() - startTime << std::endl;
    std::cout << "all: " << clock() - startTime2 << std::endl;
    vector<map<string, Point>> pointDictList = { ColumnKeyPoints, RowPointsOne, RowPointsTwo };
    for (auto& pointDict : pointDictList) {

        for (auto& item : pointDict) {
            int a = 1;
            cv::circle(img_bgr, item.second, 5, cv::Scalar(0, 0, 255), -1);

        }

    }
    cv::circle(img_bgr, point, 5, cv::Scalar(0, 255, 255), -1);
    //cv::circle(img_bgr, pointb, 5, cv::Scalar(0, 255, 255), -1);
    // 直线 white top
    float k, b;
    k = LineWhiteTop[1] / LineWhiteTop[0];
    b = LineWhiteTop[3] - k * LineWhiteTop[2];
    line(img_bgr, Point(0, (int)b), Point(600, (int)(k * 600 + b)), Scalar(255, 255, 0), 1);
    for (auto& item : ColWhiteTopPointsList) {
        int a = 1;
        cv::circle(img_bgr, item, 5, cv::Scalar(0, 0, 255), 1);

    }
    cout << k << " " << b << endl;
    // 直线
    k = output[1] / output[0];
    b = output[3] - k * output[2];
    line(img_bgr, Point(0, (int)b), Point(600, (int)(k * 600 + b)), Scalar(255, 255, 0), 1);
    cout << k << " " << b << endl;

    int a = 1;
    cv::imshow("img", img_bgr);
    cv::waitKey(0);
}


