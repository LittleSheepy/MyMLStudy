/*
2023年5月5日
*/
#include "pch.h"
#include <fstream>
#include <sys/stat.h>
#include <direct.h>

#include "CPostProcessor.h"


#define PP_DEBUG 0

#ifdef PP_DEBUG
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14标准引入的文件系统库
#include <sys/stat.h>
namespace fs = std::experimental::filesystem;
//string img_save_path_str = "D:/00myGitHub/00MyMLStudy/12CPPopenCV/bin/img_save/";
string PostProcess_img_save_path = "./img_save/";
string PostProcessAI_para = PostProcess_img_save_path + "AI_para/";
string PostProcessDebug = PostProcess_img_save_path + "PostProcessDebug/";
map<string, cv::Mat> PostProcess_map_img;
#endif // PP_DEBUG

using namespace std;

// 配置 1c1 图1 c 中间的竖支撑 编号1
#define CTOP 700
#define CBOTTOM 1630
#define BTOP 1600
#define BBOTTOM 1800
#define PIX_H   1233                        // 
#define MM_H   90
#define PIX_MM  (PIX_H/MM_H)                // 13.7
#define AREA25  int(PIX_MM*PIX_MM*25)
#define AREA150  int(PIX_MM*PIX_MM*150)     // 32320
#define LENGTH5  int(PIX_MM*5)              // 68.5
#define LENGTH30  int(PIX_MM*30)            // 411
#define CEMIAN_BROKEN ""


//vector<vector<string>> = { vector<string>{"1c1"} };
// 12(3)45(6)(7)89（10）11 12
// 中间分组 broken : 
// cb1:3 cb2:6 cb3:7 cb4:10 cb5:1245 cb6:89 11 12
//      解释：cb1: center broken 1
// 底部分组 broken : bbl bbc bbr
//      解释：bbl:bottom broken left
// 
CPostProcessor::CPostProcessor() {
    m_brokenCfg = {
        {"cb1",0},{"cb2",0},{"cb3",0},{"cb4",0},{"cb5",1},{"cb6",1},
        {"bbl",1},{"bbc",0},{"bbr",1},
    };

    m_s_g_c = { "cb5", "cb5", "cb1", "cb5", "cb5", "cb2", "cb3", "cb6", "cb6", "cb4", "cb4", "cb6" };
    m_limit_c = { AREA150, AREA150, 0, AREA150, AREA150, 0, 0, AREA150, AREA150, 0, 0, AREA150 };
    m_s_g_b = { "bbl", "bbl", "bbc", "bbr", "bbr" };
    m_limit_b = { AREA25, AREA150, 0, AREA150, AREA25 };
    img_template = cv::imread(template_path);
}

int CPostProcessor::getLimit(string bc, int ser) {
    return 0;
}

void CPostProcessor::imgCfgInit() {
    m_img1Cfg = {
        // 455-700 770-1000 1000-1300 1900-2300
        {"1c1", "cb5", 1,cv::Point(450,CTOP),cv::Point(700,CBOTTOM),AREA150},       // cb5 b:broken
        {"1c2", "cb5", 2,cv::Point(700,CTOP),cv::Point(1000,CBOTTOM),AREA150},
        {"1c3", "cb1", 3,cv::Point(1000,CTOP),cv::Point(1430,CBOTTOM)},
        {"1c4", "cb5", 4,cv::Point(1900,CTOP),cv::Point(2250,CBOTTOM),AREA150},

        {"1b6", "bbl", 1,cv::Point(450,BTOP),cv::Point(1430,BBOTTOM),AREA25},
        {"1b8", "bbl", 2,cv::Point(1430,BTOP),cv::Point(2250,BBOTTOM),AREA150},
    };
    m_img2Cfg = {
        {"2c5", "cb5", 5,cv::Point(600,CTOP),cv::Point(1200,CBOTTOM),AREA150},
        {"2c6", "cb2", 6,cv::Point(1200,CTOP),cv::Point(1800,CBOTTOM)},

        {"2b1", "bbc", 3,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img3Cfg = {
        {"3c7", "cb3", 7,cv::Point(300,CTOP),cv::Point(900,CBOTTOM)},
        {"3c8", "cb6", 8,cv::Point(1000,CTOP),cv::Point(1600,CBOTTOM),AREA150},

        {"3b3", "bbc", 3,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img4Cfg = {
        {"4c9", "cb6", 9,cv::Point(100,CTOP),cv::Point(600,CBOTTOM),AREA150},
        {"4c10", "cb4", 10,cv::Point(1100,CTOP),cv::Point(1400,CBOTTOM)},
        {"4c11", "cb6", 11,cv::Point(1400,CTOP),cv::Point(1700,CBOTTOM),AREA150},
        {"4c12", "cb6", 12,cv::Point(1800,CTOP),cv::Point(2100,CBOTTOM),AREA150},

        {"4b9", "bbr", 4,cv::Point(100,BTOP),cv::Point(1400,BBOTTOM),AREA150},
        {"4b10", "bbr", 5,cv::Point(1100,BTOP),cv::Point(2100,BBOTTOM),AREA25},
    };
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}
/*
// 0506之前的右边
void CPostProcessor::imgCfgInitByOffSet() {
    m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(380, CTOP), cv::Point(680, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(800, CTOP), cv::Point(980, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(1090, CTOP), cv::Point(1250, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(1980, CTOP), cv::Point(2210, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 1, cv::Point(430, BTOP), cv::Point(1150, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 2, cv::Point(1200, BTOP), cv::Point(2030, BBOTTOM), AREA150),
    };
    m_img2Cfg = {
        CBox("2c4", "cb5", 4, cv::Point(0, CTOP), cv::Point(320, CBOTTOM), AREA150),
        CBox("2c5", "cb5", 5, cv::Point(810, CTOP), cv::Point(1180, CBOTTOM), AREA150),
        CBox("2c6", "cb2", 6, cv::Point(1610, CTOP), cv::Point(1900, CBOTTOM)),

        CBox("2b1", "bbc", 3, cv::Point(100, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(400, CTOP), cv::Point(710, CBOTTOM)),
        CBox("3c8", "cb6", 8, cv::Point(1220, CTOP), cv::Point(1490, CBOTTOM), AREA150),
        CBox("3c9", "cb6", 9, cv::Point(2240, CTOP), cv::Point(2448, CBOTTOM), AREA150),

        CBox("3b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(2220, BBOTTOM))
    };
    m_img4Cfg = {
        CBox("4c9", "cb6", 9, cv::Point(330, CTOP), cv::Point(550, CBOTTOM), AREA150),
        CBox("4c10", "cb4", 10, cv::Point(1260, CTOP), cv::Point(1400, CBOTTOM)),
        CBox("4c11", "cb6", 11, cv::Point(1510, CTOP), cv::Point(1690, CBOTTOM), AREA150),
        CBox("4c12", "cb6", 12, cv::Point(1820, CTOP), cv::Point(2100, CBOTTOM), AREA150),

        CBox("4b9", "bbr", 4, cv::Point(500, BTOP), cv::Point(1310, BBOTTOM), AREA150),
        CBox("4b10", "bbr", 5, cv::Point(1370, BTOP), cv::Point(2050, BBOTTOM), AREA25),
    };
    m_imgCfg.clear();
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}

void CPostProcessor::imgCfgInitByOffSet2() {
    m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(550, CTOP), cv::Point(850, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(950, CTOP), cv::Point(1130, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(1230, CTOP), cv::Point(1390, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(2120, CTOP), cv::Point(2400, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 1, cv::Point(660, BTOP), cv::Point(1280, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 2, cv::Point(1340, BTOP), cv::Point(2170, BBOTTOM), AREA150),
    };
    m_img2Cfg = {
        CBox("2c4", "cb5", 4, cv::Point(220, CTOP), cv::Point(450, CBOTTOM), AREA150),
        CBox("2c5", "cb5", 5, cv::Point(1000, CTOP), cv::Point(1300, CBOTTOM), AREA150),
        CBox("2c6", "cb2", 6, cv::Point(1800, CTOP), cv::Point(2100, CBOTTOM)),

        CBox("2b1", "bbc", 3, cv::Point(280, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(600, CTOP), cv::Point(900, CBOTTOM)),
        CBox("3c8", "cb6", 8, cv::Point(1400, CTOP), cv::Point(1700, CBOTTOM), AREA150),
        CBox("3c9", "cb6", 9, cv::Point(2240, CTOP), cv::Point(2448, CBOTTOM), AREA150),

        CBox("3b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(2400, BBOTTOM))
    };
    m_img4Cfg = {
        CBox("4c9", "cb6", 9, cv::Point(200, CTOP), cv::Point(450, CBOTTOM), AREA150),
        CBox("4c10", "cb4", 10, cv::Point(1180, CTOP), cv::Point(1340, CBOTTOM)),
        CBox("4c11", "cb6", 11, cv::Point(1430, CTOP), cv::Point(1620, CBOTTOM), AREA150),
        CBox("4c12", "cb6", 12, cv::Point(1740, CTOP), cv::Point(2030, CBOTTOM), AREA150),

        //CBox("4b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(270, BBOTTOM)),
        CBox("4b9", "bbr", 4, cv::Point(400, BTOP), cv::Point(1230, BBOTTOM), AREA150),
        CBox("4b10", "bbr", 5, cv::Point(1290, BTOP), cv::Point(1980, BBOTTOM), AREA25),
    };
    m_imgCfg.clear();
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}
*/
// 左边线 0506  底部没
void CPostProcessor::imgCfgInitByOffSet() {
    m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(210, CTOP), cv::Point(520, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(620, CTOP), cv::Point(820, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(910, CTOP), cv::Point(1080, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(1820, CTOP), cv::Point(2070, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 1, cv::Point(430, BTOP), cv::Point(1150, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 2, cv::Point(1200, BTOP), cv::Point(2030, BBOTTOM), AREA150),
    };
    m_img2Cfg = {
        //CBox("2c4", "cb5", 4, cv::Point(0, CTOP), cv::Point(320, CBOTTOM), AREA150),
        CBox("2c5", "cb5", 5, cv::Point(400, CTOP), cv::Point(1100, CBOTTOM), AREA150),
        CBox("2c6", "cb2", 6, cv::Point(1200, CTOP), cv::Point(2000, CBOTTOM)),

        CBox("2b1", "bbc", 3, cv::Point(100, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(300, CTOP), cv::Point(1100, CBOTTOM)),
        CBox("3c8", "cb6", 8, cv::Point(1200, CTOP), cv::Point(1900, CBOTTOM), AREA150),
        //CBox("3c9", "cb6", 9, cv::Point(2240, CTOP), cv::Point(2448, CBOTTOM), AREA150),

        CBox("3b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(2220, BBOTTOM))
    };
    m_img4Cfg = {
        CBox("4c9", "cb6", 9, cv::Point(200, CTOP), cv::Point(700, CBOTTOM), AREA150),
        CBox("4c10", "cb4", 10, cv::Point(1300, CTOP), cv::Point(1590, CBOTTOM)),
        CBox("4c11", "cb6", 11, cv::Point(1600, CTOP), cv::Point(1800, CBOTTOM), AREA150),
        CBox("4c12", "cb6", 12, cv::Point(1830, CTOP), cv::Point(2300, CBOTTOM), AREA150),

        CBox("4b9", "bbr", 4, cv::Point(500, BTOP), cv::Point(1310, BBOTTOM), AREA150),
        CBox("4b10", "bbr", 5, cv::Point(1370, BTOP), cv::Point(2050, BBOTTOM), AREA25),
    };
    m_imgCfg.clear();
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}

// 左边线 0506  底部没
void CPostProcessor::imgCfgInitByOffSet2() {
    m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(270, CTOP), cv::Point(560, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(660, CTOP), cv::Point(850, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(930, CTOP), cv::Point(1090, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(1790, CTOP), cv::Point(2010, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 1, cv::Point(660, BTOP), cv::Point(1280, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 2, cv::Point(1340, BTOP), cv::Point(2170, BBOTTOM), AREA150),
    };
    m_img2Cfg = {
        //CBox("2c4", "cb5", 4, cv::Point(0, CTOP), cv::Point(320, CBOTTOM), AREA150),
        CBox("2c5", "cb5", 5, cv::Point(560, CTOP), cv::Point(1230, CBOTTOM), AREA150),
        CBox("2c6", "cb2", 6, cv::Point(1330, CTOP), cv::Point(2030, CBOTTOM)),

        CBox("2b1", "bbc", 3, cv::Point(100, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(240, CTOP), cv::Point(940, CBOTTOM)),
        CBox("3c8", "cb6", 8, cv::Point(1030, CTOP), cv::Point(1700, CBOTTOM), AREA150),
        CBox("3c9", "cb6", 9, cv::Point(1760, CTOP), cv::Point(2400, CBOTTOM), AREA150),

        CBox("3b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(2220, BBOTTOM))
    };
    m_img4Cfg = {
        CBox("4c9", "cb6", 9, cv::Point(200, CTOP), cv::Point(500, CBOTTOM), AREA150),
        CBox("4c10", "cb4", 10, cv::Point(1100, CTOP), cv::Point(1350, CBOTTOM)),
        CBox("4c11", "cb6", 11, cv::Point(1350, CTOP), cv::Point(1660, CBOTTOM), AREA150),
        CBox("4c12", "cb6", 12, cv::Point(1670, CTOP), cv::Point(2200, CBOTTOM), AREA150),

        //CBox("4b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(270, BBOTTOM)),
        CBox("4b9", "bbr", 4, cv::Point(400, BTOP), cv::Point(1230, BBOTTOM), AREA150),
        CBox("4b10", "bbr", 5, cv::Point(1290, BTOP), cv::Point(1980, BBOTTOM), AREA25),
    };
    m_imgCfg.clear();
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}

cv::Mat CPostProcessor::getMask(vector<cv::Point> points) {
    cv::Mat mask;

    return mask;
}
map<string, cv::Point> CPostProcessor::getRowWhitePoint(const cv::Mat& img_gray, int point_y) {
    int w = img_gray.cols;
    map<string, cv::Point> RowPoints;
    for (int point_x = 0; point_x < w; point_x++) {
        if (img_gray.at<uchar>(point_y, point_x) > 240) {
            RowPoints["left"] = cv::Point(point_x, point_y);
            break;
        }
    }
    for (int point_x = w - 1; point_x > 0; point_x--) {
        if (img_gray.at<uchar>(point_y, point_x) > 240) {
            RowPoints["right"] = cv::Point(point_x, point_y);
            break;
        }
    }
    return RowPoints;
}
cv::Point CPostProcessor::findWhiteAreaByTemplate(const cv::Mat& img_bgr) {
    int method = cv::TM_SQDIFF_NORMED;
    cv::Mat result;
    cv::matchTemplate(img_bgr, img_template, result, method);
    cv::normalize(result, result, 0, 1, cv::NORM_MINMAX, -1);
    double _minVal, _maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &_minVal, &_maxVal, &minLoc, &maxLoc, cv::Mat());
    cv::Point matchLoc;
    if (method == cv::TM_SQDIFF || method == cv::TM_SQDIFF_NORMED) {
        matchLoc = minLoc;
    }
    else {
        matchLoc = maxLoc;
    }
    return matchLoc;
}
void CPostProcessor::setOffSet(cv::Mat img_bgr, int camera_num) {
    char buf[128];
    sprintf_s(buf, "<<<<setOffSet>>> setOffSet enter>>>>>>>>>>>>>>camera_num=%d", camera_num);
    OutputDebugStringA(buf);
    // 0506之前右边线
    //if (camera_num == 0) {
    //    template_x = 1270;
    //}
    //else {
    //    template_x = 1410;
    //}
    // 0506左边线
    if (camera_num == 0) {
        template_x = 1100;
    }
    else {
        template_x = 1110;
    }
    //sprintf_s(buf, "ssss");
    //OutputDebugStringA(buf);
    //cv::Point matchLoc = findWhiteArea(img_bgr);
    cv::Mat img_gray;
    cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);
    double startTime = clock();//计时开始
    cv::Rect matchLoc = findWhiteArea(img_gray);
    cout << "findWhiteArea: " << clock() - startTime << endl;
    //startTime = clock();//计时开始
    //map<string, cv::Point> points = getRowWhitePoint(img_gray, 1300);
    //cout << "getRowWhitePoint: " << clock() - startTime << endl;
    //cout << matchLoc.x << endl;
    //cout << points["left"].x << endl;

    offset = matchLoc.x - template_x;
    if (abs(offset) > 250) {
        template_x = 1260;
        sprintf_s(buf, "[setOffSet][warning]offset=%d, white_left=%d, template_x=%d", offset, matchLoc.x, template_x);
        OutputDebugStringA(buf);
        offset = 0;
        imgCfgInit();
    }
    else {
        if (camera_num == 0) {
            imgCfgInitByOffSet();
        }
        else {
            imgCfgInitByOffSet2();
        }
    }
    OutputDebugStringA("<<<setOffSet>>> setOffSet out <<<<<<<<<<<<<<<");
}

void CPostProcessor::savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect) {
    // 创建或者清空文件夹
    static int num = 1;
    const std::string folder_path = "./img_save/AI_para";
    for (int i = 0; i < 4; i++) {
        // 保存图片
        cv::Mat img = v_img[i];
        if (img.empty()) {
            continue;
        }
        std::string filename = folder_path + "/" + std::to_string(num);
        filename = filename + "_" + std::to_string(i);
        std::string img_name = filename + ".bmp";
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

bool CPostProcessor::Process_old(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num) {
    sprintf_alg("[二次复判][Begin] camera_num=%d", camera_num);
    char buf[128];
    // 重置过程变量
    reset();
    m_CenterDefectMatched.clear();
    m_BottomDefectMatched.clear();
#ifdef PP_DEBUG
    savePara(v_img, vv_defect);
#endif // PP_DEBUG

    bool result = true;
    // 设置offset
    setOffSet(v_img[0], camera_num);
    for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
        (*it).second = 0;
    }
    vector<vector<CDefect>> vv_defect_others;
    // 遍历4个图 
    for (int i = 0; i < 4; i++) {
        cv::Mat img = v_img[i];
        vector<CDefect> v_defect_others;
        vector<CDefect> v_defect = vv_defect[i];
        for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
            if ((*it).area > 0) {
                if ((*it).type == 11) {
                    bool matched = processImg(img, (*it), i);
                    if (matched == false) {
                        v_defect_others.push_back(*it);
                    }
                }
                else {
                    sprintf_alg("[warring] type=%d name=%s", (*it).type, (*it).name);
                }
            }
            else {
                sprintf_alg("[warring] area=%d", (*it).area);
            }
        }
        vv_defect_others.push_back(v_defect_others);
    }

    // 如果有其他位置破损 确认是NG
    if (result == true) {
        for (int i = 0; i < 4; i++) {
            vector<CDefect> v_defect_others = vv_defect_others[i];
            if (v_defect_others.size() > 0) {
                sprintf_alg("[Process] rejudge is NG, img_num=%d,have other broken defect!", i);
                result = false;
                break;
            }
        }
    }
    // 遍历m_brokenCnt 确认 NG
    if (result == true) {
        for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
            string key = (*it).first;
            int val = (*it).second;
            if (val > m_brokenCfg[key]) {
                sprintf_alg("[Process] rejudge is NG, key=%s,val=%d,m_brokenCfg=%d,num is too big!", (*it).first.c_str(), val, m_brokenCfg[key]);
                result = false;
                break;
            }
        }
    }
#ifdef PP_DEBUG
    //Create a time_t object and get the current time
    time_t now = time(0);
    //Create a tm struct to hold the current time
    tm ltm;
    localtime_s(&ltm, &now);
    std::stringstream ss;
    ss << std::put_time(&ltm, "%Y%m%d%H%M");
    std::string str_time = ss.str();
    string m_brokenCnt_file = PostProcessDebug + str_time + "m_brokenCnt.txt";
    std::ofstream outputFile(m_brokenCnt_file);
    if (outputFile.is_open()) {
        // write the keys and values to the file
        for (const auto& pair : m_brokenCnt) {
            std::string key = pair.first;
            int value = pair.second;
            outputFile << key << " " << value << "\n";
            sprintf_s(buf, "<<Process>> result : %s : %d", key.c_str(), value);
            OutputDebugStringA(buf);
        }

        // 保存结果
        outputFile << "\nresult:" << " " << result << "\n";
        // close the file
        outputFile.close();
    }
#endif // PP_DEBUG
    sprintf_alg("[二次复判][End] result=%s", result ? "true" : "false");
    return result;
}

bool CPostProcessor::processImg(cv::Mat img, CDefect defect, int serial) {
    char buf[256];
    cv::Mat img_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, cv::Scalar(1), -1, 4);
    int defect_area = defect.area;
    bool result = false;
    for (auto it = m_imgCfg[serial].begin(); it != m_imgCfg[serial].end(); ++it) {
        string arr_name = (*it).arr_name;
        int cfg_area = (*it).area;
        int defect_w = abs(defect.p1.x - defect.p2.x);
        int defect_h = abs(defect.p1.y - defect.p2.y);
        // 切片
        int x1 = (*it).p1.x + offset;
        int x2 = (*it).p2.x + offset;
        if (x1 < 0) {
            x1 = 0;
        }
        if (x2 < 0) {
            x2 = 0;
        }
        if (x1 > img.cols) {
            x1 = img.cols;
        }
        if (x2 > img.cols) {
            x2 = img.cols;
        }
        cv::Rect select = cv::Rect(cv::Point(x1, (*it).p1.y), cv::Point(x2, (*it).p2.y));
        cv::Mat ROI = img_mask(select);
        double sum = cv::sum(ROI)[0];
        sprintf_s(buf, "<<processImg>> img=%d, name=%s, sum=%.3f, defect_area*0.9=%d, defect_area=%d, defect_w=%d, defect_h=%d, m_brokenCnt=%d", serial, (*it).name.c_str(), sum, (int)(defect.area * 0.7), defect.area, defect_w, defect_h, m_brokenCnt[(*it).arr_name]);
        OutputDebugStringA(buf);
        // 一多半在这个配置框就认为是这个的
        if (sum > defect.area * 0.7) {
            // 条件限制
            int lenth_ = 0;
            // 面积超限 算两个
            if (cfg_area == AREA25) {
                lenth_ = LENGTH5;
            }
            if (cfg_area == AREA150) {
                lenth_ = LENGTH30;
            }
            // 缺陷特征
            int defect_length = 0;
            if ((*it).arr_name[0] == 'c') {
                defect_length = defect_h;
            }

            if ((*it).arr_name[0] == 'b') {
                defect_length = defect_w;
            }
            sprintf_s(buf, "<<processImg>>defect_length=%d, lenth_=%d", defect_length, lenth_);
            OutputDebugStringA(buf);
            if (defect_length > lenth_) {
                m_brokenCnt[(*it).arr_name]++;
            }
            m_brokenCnt[(*it).arr_name]++;
            sprintf_s(buf, "<<processImg>>m_brokenCnt=%d", m_brokenCnt[(*it).arr_name]);
            OutputDebugStringA(buf);
            result = true;
        }
    }
    return result;
}

// BBOX分组 重叠就分为一组
vector<vector<CDefect>> CPostProcessor::groupBBoxes(vector<CDefect> bboxes) {
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

int CPostProcessor::HeBing(int serial, char bc) {
    sprintf_alg("[HeBing][enter] serial=%d, bc=%c", serial, bc);
    vector<CDefect> v_defect1 = m_CenterDefectMatched[serial - 1];
    int defect_length_index = 1;
    int cfg_area = m_limit_c[serial - 1];
    int lenth_limit = 0;
    if (cfg_area == AREA25) {
        lenth_limit = LENGTH5;
    }
    if (cfg_area == AREA150) {
        lenth_limit = LENGTH30;
    }
    //if (bc == 'b') {
    //    v_defect1 = m_BottomDefectMatched[serial - 1];
    //    defect_length_index = 0;
    //    cfg_area = m_limit_b[serial - 1];
    //}
    // 没有缺陷
    if (v_defect1.size() == 0) {
        sprintf_alg("[HeBing][out] no defect");
        return 0;
    }

    vector<vector<CDefect>> vv_defect1;
    // 合并破损框
    vv_defect1 = groupBBoxes(v_defect1);
    //int vv_defect1_size = (int)vv_defect1.size();
    //if (vv_defect1_size > 1) {
    //    sprintf_alg("[HeBing][out] vv_defect1_size > 1. vv_defect1_size=%d", vv_defect1_size);
    //    return vv_defect1_size;
    //}
    // 遍历每组破损框
    int defect_cnt = 0;
    for (vector<CDefect> v_defect : vv_defect1) {
        vector<int> resultWH = getGroupBBoxesWH(v_defect);
        int defect_length = resultWH[defect_length_index];
        if (defect_length > lenth_limit) {
            defect_cnt = 1;
            break;
        }
    }
    sprintf_alg("[HeBing][out] defect_cnt=%d", defect_cnt);
    return defect_cnt;
}
bool CPostProcessor::Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect, int camera_num) {
    sprintf_alg("[ReJudge][Begin] camera_num=%d ", camera_num);
    // 重置过程变量
    reset();
    m_CenterDefectMatched.clear();
    m_BottomDefectMatched.clear();
#ifdef PP_DEBUG
    savePara(v_img, vv_defect);
#endif // PP_DEBUG

    bool result = true;
    // 设置offset
    setOffSet(v_img[0], camera_num);
    for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
        (*it).second = 0;
    }
    vector<vector<CDefect>> vv_defect_others;
    // 遍历4个缺陷匹配置框
    for (int i = 0; i < 4; i++) {
        cv::Mat img = v_img[i];
        vector<CDefect> v_defect = vv_defect[i];
        vv_defect_others.push_back({});
        sprintf_alg("[Process] img %d v_defect.size=%d", i, v_defect.size());
        if (v_defect.size() == 0) {
            sprintf_alg("[Process][import] img %d have None defect.", i);
            continue;
        }
        for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
            //sprintf_alg("[Process]      defect_area=%d, defect_type = %d.", (*it).area, (*it).type);
            if ((*it).area > 0) {
                if ((*it).type == 11) {
                    bool matched = defectMatchBox(img, *it, i);
                    if (matched == false) {
                        // TODO 这里可以直接返回false
                        // result = false;
                        // break;
                        sprintf_alg("[Process] match failed: defect_area=%d", (*it).area);
                        vv_defect_others[i].push_back(*it);
                    }
                }
                else {
                    sprintf_alg("[Process][warring] type=%d name=%s", (*it).type, (*it).name);
                }
            }
            else {
                sprintf_alg("[Process][warring] area=%d", (*it).area);
            }
        }
        //sprintf_alg("[Process]      v_defect_others size = %d", vv_defect_others[i].size());
    }
    // 如果有其他位置破损 确认是NG
    if (result == true) {
        for (int i = 0; i < 4; i++) {
            vector<CDefect> v_defect_others = vv_defect_others[i];
            if (v_defect_others.size() > 0) {
                sprintf_alg("[Process] rejudge is NG, img_num=%d,have other broken defect!", i);
                result = false;
                break;
            }
        }
    }

    // 中间
    for (int i = 1; i < 13; ++i) {
        string group_str = m_s_g_c[i - 1];
        sprintf_alg("[Process] c group_str=%s serial=%d", group_str.c_str(), i);
        int cnt = HeBing(i, 'c');
        m_brokenCnt[group_str] += cnt;
        sprintf_alg("[Process] c m_brokenCnt group_str=%s, %d", group_str.c_str(), m_brokenCnt[group_str]);
    }
    // 底部
    //for (int i = 1; i < 6; ++i) {
    //    string group_str = m_s_g_b[i - 1];
    //    sprintf_alg("[Process] b group_str=%s", group_str.c_str());
    //    int cnt = HeBing(i, 'b');
    //    m_brokenCnt[group_str] += cnt;
    //    sprintf_alg("[Process] b m_brokenCnt group_str=%s, %d", group_str.c_str(), m_brokenCnt[group_str]);
    //}

    // 遍历m_brokenCnt 确认 NG
    if (result == true) {
        for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
            string key = (*it).first;
            int val = (*it).second;
            if (val > m_brokenCfg[key]) {
                sprintf_alg("[Process] rejudge is NG, key=%s,val=%d,m_brokenCfg=%d,num is too big!", (*it).first.c_str(), val, m_brokenCfg[key]);
                result = false;
                break;
            }
        }
    }
#ifdef PP_DEBUG
    //Create a time_t object and get the current time
    time_t now = time(0);
    //Create a tm struct to hold the current time
    tm ltm;
    localtime_s(&ltm, &now);
    std::stringstream ss;
    ss << std::put_time(&ltm, "%Y%m%d%H%M");
    std::string str_time = ss.str();
    string m_brokenCnt_file = PostProcessDebug + str_time + "m_brokenCnt.txt";
    std::ofstream outputFile(m_brokenCnt_file);
    if (outputFile.is_open()) {
        // write the keys and values to the file
        for (const auto& pair : m_brokenCnt) {
            std::string key = pair.first;
            int value = pair.second;
            outputFile << key << " " << value << "\n";
            sprintf_alg("<<Process>> result : %s : %d, , limit: %d", key.c_str(), value, m_brokenCfg[key]);
        }

        // 保存结果
        outputFile << "\nresult:" << " " << result << "\n";
        // close the file
        outputFile.close();
    }
#endif // PP_DEBUG
    sprintf_alg("[ReJudge][End] result=%s", result ? "true" : "false");
    return result;
}

bool CPostProcessor::defectMatchBox(cv::Mat img, CDefect defect, int serial) {
    //sprintf_alg("[defectMatchBox][Enter] img serial=%d, defect aera", serial, defect.area);
    int result = false;

    // 创建缺陷mask
    cv::Mat img_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, cv::Scalar(1), -1, 4);

    for (auto it = m_imgCfg[serial].begin(); it != m_imgCfg[serial].end(); ++it) {
        //sprintf_alg("[defectMatchBox] m_imgCf: name=%s, arr_name=%s", (*it).name.c_str(), (*it).arr_name.c_str());
        string arr_name = (*it).arr_name;
        int cfg_area = (*it).area;
        int defect_w = abs(defect.p1.x - defect.p2.x);
        int defect_h = abs(defect.p1.y - defect.p2.y);
        int judge_area = defect_w * defect_h;
        //sprintf_alg("[defectMatchBox] defect_w=%d, defect_h=%d", defect_w, defect_h);
        // 切片
        int x1 = (*it).p1.x + offset;
        int x2 = (*it).p2.x + offset;
        if (x1 < 0) {
            x1 = 0;
        }
        if (x2 < 0) {
            x2 = 0;
        }
        if (x1 > img.cols) {
            x1 = img.cols;
        }
        if (x2 > img.cols) {
            x2 = img.cols;
        }
        cv::Rect select = cv::Rect(cv::Point(x1, (*it).p1.y), cv::Point(x2, (*it).p2.y));
        cv::Mat ROI = img_mask(select);
        double sum = cv::sum(ROI)[0];
        //sprintf_alg("[defectMatchBox] sum=%.3f,defect_x:p1.x=%d p2.x=%d config_x:x1=%d x2=%d", sum, defect.p1.x, defect.p2.x, x1, x2);
        //sprintf_alg("[defectMatchBox]       defect_area*0.7=%d, defect_area=%d, m_brokenCnt=%d", (int)(defect.area * 0.7), defect.area, m_brokenCnt[arr_name]);
        // 一多半在这个配置框就认为是这个的
        //if (sum > defect.area * 0.7) {
        if (sum > judge_area * 0.7) {
            if ((*it).arr_name[0] == 'c') {
                m_CenterDefectMatched[(*it).serial - 1].push_back(defect);
                //sprintf_alg("[defectMatchBox]       m_CenterDefectMatched center size: serial=%d, size=%d", (*it).serial, m_CenterDefectMatched[(*it).serial].size());
            }
            else {
                m_BottomDefectMatched[(*it).serial - 1].push_back(defect);
                //sprintf_alg("[defectMatchBox]       m_BottomDefectMatched bottom size: serial=%d, size=%d", (*it).serial, m_BottomDefectMatched[(*it).serial].size());
            }
            result = true;
            break;
        }
    }
    //sprintf_alg("[defectMatchBox][Out] result=%s", result ? "true" : "false");
    return result;
}

int CPostProcessor::groupBBoxes_old(vector<CDefect> bboxes, vector<CDefect>& v_defect1, char bc) {
    vector<vector<CDefect>> vv_defect1;
    vector<CDefect> v_defect1_shengyu;
    vector<CDefect> v_defect1_group_tmp;
    //vv_defect1.push_back({ v_defect1.front() });
    v_defect1_group_tmp = { v_defect1.front() };
    v_defect1.erase(v_defect1.begin());
    v_defect1_shengyu.clear();
    bool find_new = false;
    while (true) {
        for (auto it = v_defect1.begin(); it != v_defect1.end(); ++it) {
            CDefect q = *it;
            bool foundGroup = false;
            for (int j = 0; j < v_defect1_group_tmp.size(); j++) {
                CDefect v = v_defect1_group_tmp[j];
                int intersectionArea = 0;
                if (bc == 'c') {
                    //int x1 = max(q.p1.x, v.p1.x);
                    int y1 = max(q.p1.y, v.p1.y);
                    //int x2 = min(q.p2.x, v.p2.x);
                    int y2 = min(q.p2.y, v.p2.y);
                    intersectionArea = max(0, y2 - y1);
                }
                //else {
                //    //int x1 = max(q.p1.x, v.p1.x);
                //    //int y1 = max(q.p1.y, v.p1.y);
                //    //int x2 = min(q.p2.x, v.p2.x);
                //    //int y2 = min(q.p2.y, v.p2.y);
                //    intersectionArea = 0;
                //}
                if (intersectionArea > 0) {
                    v_defect1_group_tmp.push_back(q);
                    foundGroup = true;
                    find_new = true;
                    break;
                }
            }
            if (!foundGroup) {
                v_defect1_shengyu.push_back(q);
            }
        }
        if (find_new == true) {
            find_new = false;
            v_defect1 = v_defect1_shengyu;
            v_defect1_shengyu.clear();
            continue;
        }
        // 
        if (find_new == false) {
            if (v_defect1_shengyu.size() == 0) {
                sprintf_alg("[HeBing][import] have only one group");
                vv_defect1.push_back(v_defect1_group_tmp);
                break;
            }
            else {
                sprintf_alg("[HeBing][import] have more than one group");
                return 2;
            }

        }

    }
}