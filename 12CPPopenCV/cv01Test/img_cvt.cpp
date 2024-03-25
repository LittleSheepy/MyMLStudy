#include "img_cvt.h"

void img_gray2bgr(cv::Mat img)
{
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
    cv::imwrite("./gray_result.jpg", img);
    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    cv::imwrite("./bgr_result.jpg", img);
}

void img_cvtTest(void)
{
    cv::Mat img = cv::imread("./bgr.jpg");
    img_gray2bgr(img);
}