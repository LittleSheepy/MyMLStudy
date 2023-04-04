#include <opencv2/opencv.hpp>
#include "CPostProcessor.h"
using namespace std;
using namespace cv;
// 配置 1c1 图1 c 中间的竖支撑 编号1
#define CTOP 700
#define CBOTTOM 1630
#define BTOP 1500
#define BBOTTOM 1800
//vector<vector<string>> = { vector<string>{"1c1"} };
// 12(3)45(6)(7)89（10）11 12
// 中间分组 broken : 
// cb1:3 cb2:6 cb3:7 cb4:10 cb5:1245 cb6:89 11 12
//      解释：cb1: center broken 1
// 底部分组 broken : bbl bbc bbr
//      解释：bbl:bottom broken left
// 
CPostProcessor::CPostProcessor() {
    m_img1Cfg = {
        // 455-700 770-1000 1000-1300 1900-2300
        {"1c1", "cb5", 1,Point(450,CTOP),Point(700,CBOTTOM)},
        {"1c2", "cb5", 2,Point(770,CTOP),Point(1000,CBOTTOM)},
        {"1c3", "cb1", 3,Point(1000,CTOP),Point(1300,CBOTTOM)},
        {"1c4", "cb5", 4,Point(1900,CTOP),Point(2300,CBOTTOM)},

        {"1b6", "bbl", 6,Point(450,BTOP),Point(2200,BBOTTOM)},
        {"1b7", "bbl", 7,Point(450,BTOP),Point(2200,BBOTTOM)},
        {"1b8", "bbl", 8,Point(450,BTOP),Point(2200,BBOTTOM)},
        {"1b2", "bbc", 2,Point(2000,BTOP),Point(2448,BBOTTOM)},
    };
    m_img2Cfg = {
        {"2c5", "cb5", 5,Point(600,CTOP),Point(1200,CBOTTOM)},
        {"2c6", "cb2", 6,Point(1200,CTOP),Point(1800,CBOTTOM)},

        {"2b1", "bbc", 1,Point(0,BTOP),Point(2448,BBOTTOM)},
    };
    m_img3Cfg = {
        {"3c7", "cb3", 7,Point(300,CTOP),Point(900,CBOTTOM)},
        {"3c8", "cb6", 8,Point(1000,CTOP),Point(1600,CBOTTOM)},
        {"3c9", "cb6", 9,Point(2000,CTOP),Point(2448,CBOTTOM)},

        {"3b3", "bbc", 1,Point(0,BTOP),Point(2448,BBOTTOM)},
    };
    m_img4Cfg = {
        {"4c9", "cb6", 9,Point(100,CTOP),Point(600,CBOTTOM)},
        {"4c10", "cb4", 10,Point(1100,CTOP),Point(1400,CBOTTOM)},
        {"4c11", "cb6", 11,Point(1400,CTOP),Point(1700,CBOTTOM)},
        {"4c12", "cb6", 12,Point(1800,CTOP),Point(2100,CBOTTOM)},

        {"4b3", "bbc", 1,Point(0,BTOP),Point(400,BBOTTOM)},
        {"4b9", "bbr", 9,Point(300,BTOP),Point(2200,BBOTTOM)},
        {"4b10", "bbr", 10,Point(300,BTOP),Point(2200,BBOTTOM)},
        {"4b11", "bbr", 11,Point(300,BTOP),Point(2200,BBOTTOM)},
    };
    m_brokenCfg = {
        {"cb1",0},{"cb2",0},{"cb3",0},{"cb4",0},{"cb5",1},{"cb6",1},
        {"bbl",1},{"bbc",0},{"bbr",1},
    };
}

Mat CPostProcessor::getMask(vector<Point> points) {
    Mat mask;

    return mask;
}

bool CPostProcessor::Process(vector<Mat> v_img, vector<vector<CDefect>> vv_defect) {
    bool result = true;
    // 遍历4个图
    for (int i = 0; i < 4; i++) {
        Mat img = v_img[i];
        vector<CDefect> v_defect = vv_defect[i];
        for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
            processImg1(img, (*it), i + 1);
        }
    }

    // 遍历m_brokenCnt 确认 NG
    for (auto [key, val] : m_brokenCnt) {
        if (val > m_brokenCfg[key]) {
            result = false;
            break;
        }
    }
    return result;
}
void findVerticalLine(Mat image, Point point1, Point point2) {
    Mat roi = image(Rect(point1, point2));
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);
    Mat edges;
    Canny(gray, edges, 50, 200);
    vector<Vec4i> lines;
    HoughLinesP(edges, lines, 1, CV_PI / 180, 50, 50, 10);
    for (size_t i = 0; i < lines.size(); i++) {
        Vec4i l = lines[i];
        double angle = atan2(l[3] - l[1], l[2] - l[0]) * 180 / CV_PI;
        if (abs(angle) < 100 || abs(angle) > 80) {
            line(roi, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
        }
    }
    imshow("Vertical Line", roi);
    waitKey(0);
}
void CPostProcessor::processImg1(Mat img, CDefect defect, int serial) {
    Mat img_mask = Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, Scalar(1), -1, 4);
    int area = defect.area;
    for (auto it = m_img1Cfg.begin(); it != m_img1Cfg.end(); ++it) {
        // 切片
        //findVerticalLine(img, (*it).p1, (*it).p2);
        Rect select = Rect((*it).p1, (*it).p2);
        Mat ROI = img_mask(select);
        int sum = cv::sum(ROI)[0];
        std::cout << sum << std::endl;
        if (sum > 100){
            (*it).state = false;
            (*it).n_defect++;
            m_brokenCnt[(*it).arr_name]++;
        }
    }
}