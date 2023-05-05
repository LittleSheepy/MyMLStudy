#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv.hpp>
#include <json/json.h>
using namespace std;

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

class CBox
{
public:
    string name;
    string arr_name;
    int serial;             // 编号
    cv::Point p1;
    cv::Point p2;
    int	area;
    CBox(string name, string arr_name, int serial, cv::Point p1, cv::Point p2, int	area = 0) :
        name(name), arr_name(arr_name), serial(serial), p1(p1), p2(p2), area(area) {}
};

void saveJson() {

    vector<vector<CBox>>	    m_imgCfg;
    vector<CBox> m_img1Cfg = {
        CBox("1c1", "cb5", 1, cv::Point(380, CTOP), cv::Point(680, CBOTTOM), AREA150),
        CBox("1c2", "cb5", 2, cv::Point(800, CTOP), cv::Point(980, CBOTTOM), AREA150),
        CBox("1c3", "cb1", 3, cv::Point(1090, CTOP), cv::Point(1250, CBOTTOM)),
        CBox("1c4", "cb5", 4, cv::Point(1980, CTOP), cv::Point(2210, CBOTTOM), AREA150),

        CBox("1b6", "bbl", 1, cv::Point(430, BTOP), cv::Point(1150, BBOTTOM), AREA25),
        CBox("1b8", "bbl", 2, cv::Point(1200, BTOP), cv::Point(2030, BBOTTOM), AREA150),
    };
    vector<CBox> m_img2Cfg = {
        CBox("2c4", "cb5", 4, cv::Point(0, CTOP), cv::Point(320, CBOTTOM), AREA150),
        CBox("2c5", "cb5", 5, cv::Point(810, CTOP), cv::Point(1180, CBOTTOM), AREA150),
        CBox("2c6", "cb2", 6, cv::Point(1610, CTOP), cv::Point(1900, CBOTTOM)),

        CBox("2b1", "bbc", 3, cv::Point(100, BTOP), cv::Point(2448, BBOTTOM))
    };
    vector<CBox> m_img3Cfg = {
        CBox("3c7", "cb3", 7, cv::Point(400, CTOP), cv::Point(710, CBOTTOM)),
        CBox("3c8", "cb6", 8, cv::Point(1220, CTOP), cv::Point(1490, CBOTTOM), AREA150),
        CBox("3c9", "cb6", 9, cv::Point(2240, CTOP), cv::Point(2448, CBOTTOM), AREA150),

        CBox("3b3", "bbc", 3, cv::Point(0, BTOP), cv::Point(2220, BBOTTOM))
    };
    vector<CBox> m_img4Cfg = {
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

    Json::Value m_imgCfgJson;
    for (const auto& imgCfg : m_imgCfg) {
        Json::Value imgCfgJson;
        for (const auto& box : imgCfg) {
            Json::Value boxJson;
            boxJson["name"] = box.name;
            boxJson["arr_name"] = box.arr_name;
            boxJson["serial"] = box.serial;
            boxJson["p1_x"] = box.p1.x;
            boxJson["p1_y"] = box.p1.y;
            boxJson["p2_x"] = box.p2.x;
            boxJson["p2_y"] = box.p2.y;
            boxJson["area"] = box.area;
            imgCfgJson.append(boxJson);
        }
        m_imgCfgJson.append(imgCfgJson);
    }

    std::ofstream ofs("m_imgCfg.json");
    ofs << m_imgCfgJson;
    ofs.close();
}