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

        {"1b6", "bbl", 1,Point(450,BTOP),Point(2200,BBOTTOM)},
        {"1b7", "bbl", 1,Point(450,BTOP),Point(2200,BBOTTOM)},
        {"1b8", "bbl", 1,Point(450,BTOP),Point(2200,BBOTTOM)},
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
        {"4b3", "bbr", 2,Point(300,BTOP),Point(2200,BBOTTOM)},
    };
    m_brokenCfg = {
        {"cb1",0},{"cb2",0},{"cb3",0},{"cb4",0},{"cb5",1},{"cb6",1},
        {"bbl",1},
    };
}

bool CPostProcessor::Process(vector<Mat> v_img) {
    bool result = true;
    int i = 0;
    for (auto it = v_img.begin(); it != v_img.end(); ++it) {
        processImg1(*it, ++i);
    }
    return result;
}

void CPostProcessor::processImg1(Mat img, int serial) {
    for (auto it = m_img1Cfg.begin(); it != m_img1Cfg.end(); ++it) {
        // 切片
        Rect select = Rect((*it).p1, (*it).p2);
        Mat ROI = img(select);
        int sum = cv::sum(ROI)[0];
        std::cout << sum << std::endl;
        if (sum > 100){
            (*it).state = false;
            m_brokenCnt[(*it).arr_name]++;
        }
    }
}