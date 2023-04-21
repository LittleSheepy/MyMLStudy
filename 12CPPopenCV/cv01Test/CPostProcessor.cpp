#include <opencv2/opencv.hpp>
#include <fstream>
#include <sys/stat.h>
#include <direct.h>
#include "CPostProcessor.h"
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <Windows.h>

#define DEBUG 0

#ifdef DEBUG
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14标准引入的文件系统库
#include <sys/stat.h>
namespace fs = std::experimental::filesystem;
//string img_save_path_str = "D:/00myGitHub/00MyMLStudy/12CPPopenCV/bin/img_save/";
string PostProcess_img_save_path = "./img_save/";
string PostProcessAI_para = PostProcess_img_save_path + "AI_para/";
string PostProcessDebug = PostProcess_img_save_path + "PostProcessDebug/";
map<string, cv::Mat> PostProcess_map_img;
#endif // DEBUG

using namespace std;

// 配置 1c1 图1 c 中间的竖支撑 编号1
#define CTOP 700
#define CBOTTOM 1630
#define BTOP 1600
#define BBOTTOM 1800
#define PIX_H   1233
#define MM_H   84
#define PIX_MM  (PIX_H/MM_H)                // 14'6785
#define AREA25  int(PIX_MM*PIX_MM*25)
#define AREA150  int(PIX_MM*PIX_MM*150)     // 32320

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
        {"1c1", "cb5", 1,cv::Point(450,CTOP),cv::Point(700,CBOTTOM),AREA150},
        {"1c2", "cb5", 2,cv::Point(700,CTOP),cv::Point(1000,CBOTTOM),AREA150},
        {"1c3", "cb1", 3,cv::Point(1000,CTOP),cv::Point(1430,CBOTTOM)},
        {"1c4", "cb5", 4,cv::Point(1900,CTOP),cv::Point(2250,CBOTTOM),AREA150},

        {"1b6", "bbl", 6,cv::Point(450,BTOP),cv::Point(1430,BBOTTOM),AREA25},
        {"1b8", "bbl", 8,cv::Point(1430,BTOP),cv::Point(2250,BBOTTOM),AREA150},
        {"1b2", "bbc", 2,cv::Point(2000,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img2Cfg = {
        {"2c5", "cb5", 5,cv::Point(600,CTOP),cv::Point(1200,CBOTTOM),AREA150},
        {"2c6", "cb2", 6,cv::Point(1200,CTOP),cv::Point(1800,CBOTTOM)},

        {"2b1", "bbc", 1,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img3Cfg = {
        {"3c7", "cb3", 7,cv::Point(300,CTOP),cv::Point(900,CBOTTOM)},
        {"3c8", "cb6", 8,cv::Point(1000,CTOP),cv::Point(1600,CBOTTOM),AREA150},
        {"3c9", "cb6", 9,cv::Point(2000,CTOP),cv::Point(2448,CBOTTOM),AREA150},

        {"3b3", "bbc", 1,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img4Cfg = {
        {"4c9", "cb6", 9,cv::Point(100,CTOP),cv::Point(600,CBOTTOM),AREA150},
        {"4c10", "cb4", 10,cv::Point(1100,CTOP),cv::Point(1400,CBOTTOM)},
        {"4c11", "cb6", 11,cv::Point(1400,CTOP),cv::Point(1700,CBOTTOM),AREA150},
        {"4c12", "cb6", 12,cv::Point(1800,CTOP),cv::Point(2100,CBOTTOM),AREA150},

        {"4b3", "bbc", 1,cv::Point(0,BTOP),cv::Point(400,BBOTTOM)},
        {"4b9", "bbr", 9,cv::Point(100,BTOP),cv::Point(1400,BBOTTOM),AREA150},
        {"4b10", "bbr", 10,cv::Point(1100,BTOP),cv::Point(2100,BBOTTOM),AREA25},
    };
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
    m_brokenCfg = {
        {"cb1",0},{"cb2",0},{"cb3",0},{"cb4",0},{"cb5",1},{"cb6",1},
        {"bbl",1},{"bbc",0},{"bbr",1},
    };
    img_template = cv::imread(template_path);
}

void CPostProcessor::imgCfgInit() {
    m_img1Cfg = {
        // 455-700 770-1000 1000-1300 1900-2300
        {"1c1", "cb5", 1,cv::Point(450,CTOP),cv::Point(700,CBOTTOM),AREA150},
        {"1c2", "cb5", 2,cv::Point(700,CTOP),cv::Point(1000,CBOTTOM),AREA150},
        {"1c3", "cb1", 3,cv::Point(1000,CTOP),cv::Point(1430,CBOTTOM)},
        {"1c4", "cb5", 4,cv::Point(1900,CTOP),cv::Point(2250,CBOTTOM),AREA150},

        {"1b6", "bbl", 6,cv::Point(450,BTOP),cv::Point(1430,BBOTTOM),AREA25},
        {"1b8", "bbl", 8,cv::Point(1430,BTOP),cv::Point(2250,BBOTTOM),AREA150},
        {"1b2", "bbc", 2,cv::Point(2000,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img2Cfg = {
        {"2c5", "cb5", 5,cv::Point(600,CTOP),cv::Point(1200,CBOTTOM),AREA150},
        {"2c6", "cb2", 6,cv::Point(1200,CTOP),cv::Point(1800,CBOTTOM)},

        {"2b1", "bbc", 1,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img3Cfg = {
        {"3c7", "cb3", 7,cv::Point(300,CTOP),cv::Point(900,CBOTTOM)},
        {"3c8", "cb6", 8,cv::Point(1000,CTOP),cv::Point(1600,CBOTTOM),AREA150},
        {"3c9", "cb6", 9,cv::Point(2000,CTOP),cv::Point(2448,CBOTTOM),AREA150},

        {"3b3", "bbc", 1,cv::Point(0,BTOP),cv::Point(2448,BBOTTOM)},
    };
    m_img4Cfg = {
        {"4c9", "cb6", 9,cv::Point(100,CTOP),cv::Point(600,CBOTTOM),AREA150},
        {"4c10", "cb4", 10,cv::Point(1100,CTOP),cv::Point(1400,CBOTTOM)},
        {"4c11", "cb6", 11,cv::Point(1400,CTOP),cv::Point(1700,CBOTTOM),AREA150},
        {"4c12", "cb6", 12,cv::Point(1800,CTOP),cv::Point(2100,CBOTTOM),AREA150},

        {"4b3", "bbc", 1,cv::Point(0,BTOP),cv::Point(400,BBOTTOM)},
        {"4b9", "bbr", 9,cv::Point(100,BTOP),cv::Point(1400,BBOTTOM),AREA150},
        {"4b10", "bbr", 10,cv::Point(1100,BTOP),cv::Point(2100,BBOTTOM),AREA25},
    };
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}
void CPostProcessor::imgCfgInitByOffSet() {
    m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(500, CTOP), cv::Point(630, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(840, CTOP), cv::Point(920, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(1110, CTOP), cv::Point(1200, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(2020, CTOP), cv::Point(2150, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 6, cv::Point(450, BTOP), cv::Point(1430, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 8, cv::Point(1430, BTOP), cv::Point(2250, BBOTTOM), AREA150),
        CBox("1b2", "bbc", 2, cv::Point(2000, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img2Cfg = {
        CBox("2c4", "cb5", 4, cv::Point(0, CTOP), cv::Point(180, CBOTTOM), AREA150),
            CBox("2c5", "cb5", 5, cv::Point(820, CTOP), cv::Point(960, CBOTTOM), AREA150),
            CBox("2c6", "cb2", 6, cv::Point(1640, CTOP), cv::Point(1800, CBOTTOM)),

            CBox("2b1", "bbc", 1, cv::Point(0, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(500, CTOP), cv::Point(700, CBOTTOM)),
            CBox("3c8", "cb6", 8, cv::Point(1300, CTOP), cv::Point(1500, CBOTTOM), AREA150),
            CBox("3c9", "cb6", 9, cv::Point(2100, CTOP), cv::Point(2300, CBOTTOM), AREA150),

            CBox("3b3", "bbc", 1, cv::Point(0, BTOP), cv::Point(2448, BBOTTOM))
    };
    m_img4Cfg = {
        CBox("4c9", "cb6", 9, cv::Point(250, CTOP), cv::Point(420, CBOTTOM), AREA150),
        CBox("4c10", "cb4", 10, cv::Point(1200, CTOP), cv::Point(1300, CBOTTOM)),
        CBox("4c11", "cb6", 11, cv::Point(1480, CTOP), cv::Point(1560, CBOTTOM), AREA150),
        CBox("4c12", "cb6", 12, cv::Point(1800, CTOP), cv::Point(2030, CBOTTOM), AREA150),

        CBox("4b3", "bbc", 1, cv::Point(0, BTOP), cv::Point(280, BBOTTOM)),
        CBox("4b9", "bbr", 9, cv::Point(380, BTOP), cv::Point(1240, BBOTTOM), AREA150),
        CBox("4b10", "bbr", 10, cv::Point(1300, BTOP), cv::Point(2030, BBOTTOM), AREA25),
    };
    m_imgCfg.push_back(m_img1Cfg);
    m_imgCfg.push_back(m_img2Cfg);
    m_imgCfg.push_back(m_img3Cfg);
    m_imgCfg.push_back(m_img4Cfg);
}
cv::Mat CPostProcessor::getMask(vector<cv::Point> points) {
    cv::Mat mask;

    return mask;
}
cv::Point CPostProcessor::findWhiteArea(cv::Mat img_bgr) {
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
void CPostProcessor::setOffSet(cv::Mat img_bgr) {
    cv::Point matchLoc = findWhiteArea(img_bgr);
    offset = matchLoc.x - template_x;
    if (abs(offset) > 250) {
        offset = 0;
        imgCfgInit();
    }
    else {
        imgCfgInitByOffSet();
    }
}

void CPostProcessor::savePara(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect) {
    //char buf[128];
    //sprintf_s(buf, "ssss");
    //OutputDebugStringA(buf);
    // 
    // 创建或者清空文件夹
    static int num = 1;
    const std::string folder_path = "./img_save/AI_para";
    //if (fs::exists(folder_path)) {
    //    // Folder exists, clear its contents
    //    for (auto& file : fs::directory_iterator(folder_path)) {
    //        fs::remove_all(file);
    //    }
    //}
    //else {
    //    // Folder does not exist, create it
    //    fs::create_directory(folder_path);
    //}
    for (int i = 0; i < 4; i++) {
        // 保存图片
        cv::Mat img = v_img[i];
        std::string filename = folder_path + "/" + std::to_string(num);
        filename = filename + "_" + std::to_string(i);
        std::string img_name = filename + ".bmp";
        cv::imwrite(img_name, img);
        // 保存 vector<CDefect>

        vector<CDefect> v_defect = vv_defect[i];
        std::string vect_name = filename + ".bin";
        std::ofstream outputFile(vect_name);

        // check if the file was successfully opened
        if (outputFile.is_open()) {
            // loop through each element in the vector and write it to the file
            for (const auto& defect : v_defect) {
                outputFile.write(reinterpret_cast<const char*>(&defect), sizeof(defect));
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
bool CPostProcessor::Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect) {
    char buf[128];
    OutputDebugStringA("<<Process>> begin Process");
#ifdef DEBUG
    savePara(v_img, vv_defect);
#endif // DEBUG

    bool result = true;
    // 设置offset
    setOffSet(v_img[0]);
    for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
        (*it).second = 0;
    }
    // 遍历4个图 
    for (int i = 0; i < 4; i++) {
        cv::Mat img = v_img[i];
        vector<CDefect> v_defect = vv_defect[i];
        for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
            if ((*it).area > 0) {
                processImg(img, (*it), i);
            }
        }
    }

    // 遍历m_brokenCnt 确认 NG
    for (auto it = m_brokenCnt.begin(); it != m_brokenCnt.end(); ++it) {
        string key = (*it).first;
        int val = (*it).second;
        if (val > m_brokenCfg[key]) {
            result = false;
            break;
        }
    }
#ifdef DEBUG
    string m_brokenCnt_file = PostProcessDebug + "m_brokenCnt.txt";
    std::ofstream outputFile(m_brokenCnt_file);
    if (outputFile.is_open()) {
        // write the keys and values to the file
        for (const auto& pair : m_brokenCnt) {
            std::string key = pair.first;
            int value = pair.second;
            outputFile << key << " " << value << "\n";
        }
        // 保存结果
        outputFile << "\nresult:" << " " << result << "\n";
        // close the file
        outputFile.close();
    }
#endif // DEBUG
    sprintf_s(buf, "<<Process>> result : %s", result ? "true" : "false");
    OutputDebugStringA(buf);
    OutputDebugStringA("<<Process>> begin Process");
    return result;
}

void CPostProcessor::processImg(cv::Mat img, CDefect defect, int serial) {
    cv::Mat img_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, cv::Scalar(1), -1, 4);
    int defect_area = defect.area;

    for (auto it = m_imgCfg[serial].begin(); it != m_imgCfg[serial].end(); ++it) {
        string arr_name = (*it).arr_name;
        int cfg_area = (*it).area;
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
        int sum = cv::sum(ROI)[0];
        // 一多半在这个配置框就认为是这个的
        if (sum > defect.area * 0.5) {
            // 面积超限 算两个
            if (defect_area > cfg_area) {
                (*it).state = false;
                (*it).n_defect++;
                m_brokenCnt[(*it).arr_name]++;
            }
            (*it).n_defect++;
            m_brokenCnt[(*it).arr_name]++;
        }
    }
}