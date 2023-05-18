/*
2023年5月11日
*/
#include "pch.h"
#include <cmath>
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14标准引入的文件系统库
namespace fs = std::experimental::filesystem;
#include <iostream>

#include "CReJudgeBack.h"

#define RJB_DEBUG 0

#define PIX_H                   145                        // 
#define MM_H                    30
#define PIX_MM                  (PIX_H/MM_H)                // 4.83
#define LEN5MM                  int(PIX_MM*5)               // 24
//#define LEN_PIX                 580
//#define W_PIX                   120
//#define EXPANSION               30
//#define LEN_HOLE                80
//#define EXPANSION_CORNER        10

CReJudgeBack::CReJudgeBack() {
    CAlgBase::loadPixAccuracyCfg();
    m_len5mm = std::round(5 / m_pix_accuracy.back);
    loadCfg();
}

bool CReJudgeBack::loadCfg() {
    if (!fs::exists("Config/ReJudgeBack.cfg")) {
        sprintf_alg("[ReJudgeBack][error] cfg file not exist!!!");
        return false;
    }
    std::ifstream ifs("Config/ReJudgeBack.cfg");
    Json::Reader reader;
    Json::Value m_imgCfgJson;
    reader.parse(ifs, m_imgCfgJson);

    m_cfgRJB.len_pix = m_imgCfgJson["LEN_PIX"].asInt();
    m_cfgRJB.w_pix = m_imgCfgJson["W_PIX"].asInt();
    m_cfgRJB.expansion = m_imgCfgJson["EXPANSION"].asInt();
    m_cfgRJB.len_hole = m_imgCfgJson["LEN_HOLE"].asInt();
    m_cfgRJB.expansion_corner = m_imgCfgJson["EXPANSION_CORNER"].asInt();
    return true;
}

void CReJudgeBack::imgCfgInit() {
    int LEN_PIX = m_cfgRJB.len_pix;
    int W_PIX = m_cfgRJB.w_pix;
    int EXPANSION = m_cfgRJB.expansion;
    int LEN_HOLE = m_cfgRJB.len_hole;
    int EXPANSION_CORNER = m_cfgRJB.expansion_corner;
    m_imgCfg = {
        {"1", "tl", 1, m_Point_lt - cv::Point(EXPANSION_CORNER,EXPANSION_CORNER), cv::Point(m_Point_lt.x + LEN_PIX,m_Point_lt.y + W_PIX),m_len5mm},
        {"2", "lt", 2, m_Point_lt - cv::Point(EXPANSION_CORNER,EXPANSION_CORNER), cv::Point(m_Point_lt.x + W_PIX,m_Point_lt.y + LEN_PIX),m_len5mm},

        {"3", "bl", 3, m_Point_lb - cv::Point(EXPANSION_CORNER,-EXPANSION_CORNER), cv::Point(m_Point_lb.x + LEN_PIX,m_Point_lb.y - W_PIX),m_len5mm},
        {"4", "lb", 4, m_Point_lb - cv::Point(EXPANSION_CORNER,-EXPANSION_CORNER), cv::Point(m_Point_lb.x + W_PIX,m_Point_lb.y - LEN_PIX),m_len5mm},

        {"5", "tr", 5, m_Point_rt - cv::Point(-EXPANSION_CORNER,EXPANSION_CORNER), cv::Point(m_Point_rt.x - LEN_PIX,m_Point_rt.y + W_PIX),m_len5mm},
        {"6", "rt", 6, m_Point_rt - cv::Point(-EXPANSION_CORNER,EXPANSION_CORNER), cv::Point(m_Point_rt.x - W_PIX,m_Point_rt.y + LEN_PIX),m_len5mm},

        {"7", "br", 7, m_Point_rb - cv::Point(-EXPANSION_CORNER,-EXPANSION_CORNER), cv::Point(m_Point_rb.x - LEN_PIX,m_Point_rb.y - W_PIX),m_len5mm},
        {"8", "rb", 8, m_Point_rb - cv::Point(-EXPANSION_CORNER,-EXPANSION_CORNER), cv::Point(m_Point_rb.x - W_PIX,m_Point_rb.y - LEN_PIX),m_len5mm},
    };
    m_imgCfgCenter = {
        {"1", "tc", 1, cv::Point(m_Point_lt.x + LEN_PIX,m_Point_lt.y - EXPANSION), cv::Point(m_Point_rt.x - LEN_PIX,m_Point_rt.y + W_PIX + LEN_HOLE + EXPANSION),0},
        {"2", "bc", 2, cv::Point(m_Point_lb.x + LEN_PIX,m_Point_lb.y + EXPANSION), cv::Point(m_Point_rb.x - LEN_PIX,m_Point_rb.y - W_PIX - LEN_HOLE - EXPANSION),0},

        {"3", "lc", 3, cv::Point(m_Point_lt.x - EXPANSION,m_Point_lt.y + LEN_PIX), cv::Point(m_Point_lb.x + W_PIX + LEN_HOLE + EXPANSION,m_Point_lb.y - LEN_PIX),0},
        {"4", "rc", 4, cv::Point(m_Point_rt.x - W_PIX - LEN_HOLE - EXPANSION,m_Point_rt.y + LEN_PIX), cv::Point(m_Point_rb.x + EXPANSION,m_Point_rb.y - LEN_PIX),0}
    };
}
bool CReJudgeBack::getPoint(cv::Mat img_gray) {
    vector<cv::Point> leftPointsList;
    vector<cv::Point> rightPointsList;
    vector<cv::Point> topPointsList;
    vector<cv::Point> bottomPointsList;
    // 获取左面的点 
    for (int i = 900; i <= 2000; i += 100) {
        cv::Point PointTmp = getLeftPoint(img_gray, i);     // 
        leftPointsList.push_back(PointTmp);
    }
    // 获取右面的点 
    for (int i = 900; i <= 2000; i += 100) {
        cv::Point PointTmp = getRightPoint(img_gray, i);     // 
        rightPointsList.push_back(PointTmp);
    }
    // 获取上面的点 
    for (int i = 1200; i <= 2800; i += 100) {
        cv::Point PointTmp = getTopPoint(img_gray, i);     // 
        topPointsList.push_back(PointTmp);
    }
    // 获取下面的点 
    for (int i = 1200; i <= 2800; i += 100) {
        cv::Point PointTmp = getBottomPoint(img_gray, i);     // 
        bottomPointsList.push_back(PointTmp);
    }
    cv::Vec4f leftLine;
    cv::Vec4f rightLine;
    cv::Vec4f topLine;
    cv::Vec4f bottomLine;
    fitLine(leftPointsList, leftLine, cv::DIST_L2, 0, 0.01, 0.01);
    fitLine(rightPointsList, rightLine, cv::DIST_L2, 0, 0.01, 0.01);
    fitLine(topPointsList, topLine, cv::DIST_L2, 0, 0.01, 0.01);
    fitLine(bottomPointsList, bottomLine, cv::DIST_L2, 0, 0.01, 0.01);
    //cv::Point2f intersection;
    //cv::intersectLines(leftLine, rightLine, intersection);
    //bool intersected = cv::intersectLines(leftLine, rightLine, m_Point_lt);
    m_Point_lt = getIntersectionPoint(leftLine, topLine);
    m_Point_lb = getIntersectionPoint(leftLine, bottomLine);
    m_Point_rt = getIntersectionPoint(rightLine, topLine);
    m_Point_rb = getIntersectionPoint(rightLine, bottomLine);
    sprintf_alg("[getPoint] m_Point_lt=%d, %d!", m_Point_lt.x, m_Point_lt.y);
    sprintf_alg("[getPoint] m_Point_lb=%d, %d!", m_Point_lb.x, m_Point_lb.y);
    sprintf_alg("[getPoint] m_Point_rt=%d, %d!", m_Point_rt.x, m_Point_rt.y);
    sprintf_alg("[getPoint] m_Point_rb=%d, %d!", m_Point_rb.x, m_Point_rb.y);
    return true;
}

void test_showcfg(cv::Mat img_mask, vector<CBox> cfg, vector<CBox> cfg2, vector<CDefect> v_defect) {
    cv::Mat img_save;
    //img_save = cv::imread("D:/02dataset/01work/05nanjingLG/06ReJudgeBack/testSimple/img.jpg");
    //img_save = cv::imread("./Config/BackRegion.jpg");
    cv::cvtColor(img_mask, img_save, cv::COLOR_GRAY2BGR);
    for (auto it : cfg) {
        cv::rectangle(img_save, it.p1, it.p2, cv::Scalar(255, 255, 0), 10);
    }
    for (auto it : cfg2) {
        cv::rectangle(img_save, it.p1, it.p2, cv::Scalar(255, 0, 255), 10);
    }
    for (auto it : v_defect) {
        cv::rectangle(img_save, it.p1, it.p2, cv::Scalar(0, 0, 255), 2);
    }
    cv::imwrite("./img_save/CReJudgeBackCfg.jpg", img_save);
}

// 返回 true是OK  false是NG
bool CReJudgeBack::Process(cv::Mat img_mask, vector<CDefect> v_defect) {
    static int num = -1;
    num++;
    char buf[256] = { 0 };
    OutputDebugStringA("[ReJudgeBack][Process][Begin]");
    sprintf_alg("[ReJudgeBack][Process][Begin]");

    std::string NGBBOX_name = "./img_save/AI_para/ReJudgeBack/触发NG条件的BBOX" + to_string(num) + ".txt";
    std::ofstream outputNGBBOXFile(NGBBOX_name);
    outputNGBBOXFile << "编号:" << num << endl;
    outputNGBBOXFile << "所有检测框:" << num << endl;
    for (const auto& defect : v_defect) {
        //outputFile.write(reinterpret_cast<const char*>(&defect), sizeof(defect));
        outputNGBBOXFile << "检测框名:" << defect.name << endl;
        outputNGBBOXFile << "左上角点:" << defect.p1.x << " " << defect.p1.y << endl;
        outputNGBBOXFile << "右下角点:" << defect.p2.x << " " << defect.p2.y << endl;
    }

    if (img_mask.empty()) {
        sprintf_alg("[Process][import] img is empty.");
        //sprintf_alg("[ReJudgeBack][Process][End] result=false");
        // 自己读图
        img_mask = cv::imread("./Config/BackRegion.jpg");

        //return false;
    }

    sprintf_alg("[Process] img channels is %d.", img_mask.channels());
    bool result = true;
    // 空图
    if (img_mask.empty()) {
        sprintf_alg("[Process][import] img is empty.");
        sprintf_alg("[ReJudgeBack][Process][End] result=false");
        return false;
    }
    // 空缺陷
    if (v_defect.size() == 0) {
        sprintf_alg("[Process][import] img have None defect.");
        sprintf_alg("[ReJudgeBack][Process][End] result=true");
        return true;
    }
    else {
        sprintf_alg("[Process][import] img v_defect.size=%d", v_defect.size());
    }
    cv::Mat img;
    if (img_mask.channels() != 1) {
        cv::Mat grayImage;
        cv::cvtColor(img_mask, img, cv::COLOR_BGR2GRAY);
        img_mask = img;
    }
    // 二值化
    cv::threshold(img_mask, img, 100, 255, cv::THRESH_BINARY);
    img_mask = img;
    // 重置过程变量
    reset();
    m_DefectMatched.clear();
    m_otherTypeDefect = false;

#ifdef RJB_DEBUG
    //savePara(img_mask, v_defect, "./img_save/AI_para/ReJudgeBack/");
#endif // RJB_DEBUG

    getPoint(img_mask);

    imgCfgInit();
    test_showcfg(img_mask, m_imgCfgCenter, m_imgCfg, v_defect);
    vector<CDefect> v_defect_others = {};
    vector<CDefect> v_defect_in_mask = {};

    // 把缺陷分为在mask上和不在mask上
    getDefectsInMask(img_mask, v_defect);

    sprintf_alg("[Process]      1 m_defectsOthers size = %d", m_defectsOthers.size());

    // 给缺陷分组
    getDefectsGroup(img_mask, m_defectsInMask);

    sprintf_alg("[Process]      2 m_defectsOthers size = %d", m_defectsOthers.size());

    // 复判缺陷 有其他缺陷
    if (v_defect_others.size() > 0) {
        result = false;
        sprintf_alg("[Process] ReJudgeFront is NG,have other broken defect!");
        // TODO 上线可以在这里直接返回false
        //sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
        //return result;
    }
    vector<CDefect> v_defect_NG;
    // 复判缺陷 边中间有缺陷
    if (m_defectsInCenter.size() > 0) {
        outputNGBBOXFile << "《《重点》》棱边中间区域有框。" << endl;
        outputNGBBOXFile << "触发NG的框:" << endl;
        outputNGBBOXFile << "《《重点》》!!!!!!!!!!!蓝圈圈出!!!!!!!!" << endl;
        CDefect defect = m_defectsInCenter[0];
        v_defect_NG.push_back(defect);
        outputNGBBOXFile << "检测框名:" << defect.name << endl;
        outputNGBBOXFile << "左上角点:" << defect.p1.x << " " << defect.p1.y << endl;
        outputNGBBOXFile << "右下角点:" << defect.p2.x << " " << defect.p2.y << endl;
        outputNGBBOXFile.close();
        result = false;
        sprintf_alg("[Process] ReJudgeFront is NG,Center have broken defect!");
        // TODO 上线可以在这里直接返回false
        //sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
        //return result;
    }

    // 判断边角矩形框缺陷个数超限制
    if (result) {
        for (const auto& pair : m_DefectGroup) {
            int key = pair.first;
            const vector<CDefect>& v_defect1 = pair.second;
            vector<vector<CDefect>> vv_defect1 = groupBBoxesByType(v_defect1);

            int defect_cnt = 0;
            vector<vector<CDefect>> vv_defect_long;
            for (vector<CDefect> v_defect2 : vv_defect1) {
                vector<int> resultWH = getGroupBBoxesWH(v_defect2);
                int defect_length = std::max(resultWH[0], resultWH[1]);
                if (defect_length > m_len5mm) {
                    vv_defect_long.push_back(v_defect2);
                    defect_cnt++;
                }
            }
            if (defect_cnt >= 2) {
                sprintf_alg("[Process] ReJudgeFront is NG,key=%d, Side defect num is too more!!!", key);
                outputNGBBOXFile << "《《重点》》边角同一组，大于5mm缺陷个数大于或等于两个！！！！！" << endl;
                outputNGBBOXFile << "触发NG的框:" << endl;
                outputNGBBOXFile << "《《重点》》!!!!!!!!!!!蓝圈圈出!!!!!!!!" << endl;
                for (const auto& v_defect_long : vv_defect_long) {
                    outputNGBBOXFile << "组开始:" << endl;
                    for (const auto& defect : v_defect_long) {
                        v_defect_NG.push_back(defect);
                        outputNGBBOXFile << "检测框名:" << defect.name << endl;
                        outputNGBBOXFile << "左上角点:" << defect.p1.x << " " << defect.p1.y << endl;
                        outputNGBBOXFile << "右下角点:" << defect.p2.x << " " << defect.p2.y << endl;
                    }
                    outputNGBBOXFile << "组结束:" << endl;
                }
                result = false;
                outputNGBBOXFile.close();
                // TODO 上线可以在这里直接返回false
                sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
                break;
            }
            else {
                sprintf_alg("[Process] ReJudgeFront is OK,key=%d, Side defect num is less, num=%d.", key, defect_cnt);
            }
        }
    }
    std::string filename_result = "./img_save/AI_para/ReJudgeBack/" + std::to_string(num);
    filename_result = filename_result + "_result";
    std::string img_name = filename_result + ".jpg";
    std::string filename_old = "./img_save/AI_para/ReJudgeBack/" + std::to_string(num);
    filename_old = filename_old + "_old";
    std::string img_name_old = filename_old + ".jpg";
    //img_mask = cv::imread("./Config/BackRegion.jpg");
    cv::Mat img_save = cv::imread("./Config/yuantu.bmp");
    cv::Mat img_save_old = cv::imread("./Config/yuantu.bmp");
    //cv::cvtColor(img_mask, img_save, cv::COLOR_GRAY2BGR);

    // 保存old
    cv::Scalar color = cv::Scalar(0, 0, 255);
    cv::putText(img_save_old, "NG", cv::Point(100, 220), cv::FONT_HERSHEY_PLAIN, 20, color, 10);

    for (auto it : v_defect) {
        _DrawDefect(img_save_old, it, 0, color);
    }

    // 保存之后的
    if (result) {
        cv::Scalar color = cv::Scalar(0, 255, 0);
        cv::putText(img_save, "OK", cv::Point(100, 220), cv::FONT_HERSHEY_PLAIN, 20, color, 10);
        for (auto it : v_defect) {
            _DrawDefect(img_save, it, 0, color);
        }
    }
    else {
        cv::Scalar color = cv::Scalar(0, 0, 255);
        cv::putText(img_save, "NG", cv::Point(100, 220), cv::FONT_HERSHEY_PLAIN, 20, color, 10);

        for (auto it : v_defect) {
            cv::Point center = (it.p1 + it.p2) / 2;
            int radius = 30;

            // Draw the circle on the image
            circle(img_save, center, radius, cv::Scalar(255, 0, 0), 3);
            _DrawDefect(img_save, it, 0, color);
        }
        //color = cv::Scalar(255, 0, 0);
        //for (auto it : v_defect_NG) {
        //    _DrawDefect(img_save, it, 0, color);
        //}
    }

    cv::imwrite(img_name, img_save);
    cv::imwrite(img_name_old, img_save_old);
    if (m_otherTypeDefect) {
        sprintf_alg("[ReJudgeBack][Process][error] Have other defect!!!");
        result = false;
    }

    sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
    sprintf_s(buf, "[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
    OutputDebugStringA(buf);
    return result;
}

bool CReJudgeBack::getDefectsInMask(cv::Mat img_mask, vector<CDefect> v_defect) {
    // 二值化
    cv::Mat img;
    cv::threshold(img_mask, img, 50, 255, cv::THRESH_BINARY);
    // 遍历
    for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
        //sprintf_alg("[Process]      defect_area=%d, defect_type = %d.", (*it).area, (*it).type);
        if ((*it).area <= 0) {
            sprintf_alg("[getDefectsInMask][warring]     defect_area is 0, area=%d", (*it).area);
            continue;
        }
        if ((*it).type != 10) {
            sprintf_alg("[getDefectsInMask][warring]type is not 10. type=%d name=%s", (*it).type, (*it).name.c_str());
            m_otherTypeDefect == true;
            continue;
        }
        bool matched = defectInMask(img, *it);
        if (matched) {
            m_defectsInMask.push_back(*it);
            sprintf_alg("[getDefectsInMask]       m_defectsInMask size : %d", m_defectsInMask.size());
        }
        else {
            // TODO 这里可以直接返回false
            // result = false;
            // break;
            sprintf_alg("[getDefectsInMask] match failed: defect_area=%d", (*it).area);
            m_defectsOthers.push_back(*it);
            sprintf_alg("[getDefectsInMask]       m_defectsOthers size : %d", m_defectsOthers.size());
        }
    }
    return true;
}
// 返回 true是在mask中 false是不在mask中
bool CReJudgeBack::defectInMask(cv::Mat img, CDefect defect) {
    sprintf_alg("");
    sprintf_alg("[defectInMask][Enter] defect aera", defect.area);
    int result = false;

    // 计算缺陷外接框面积和WH
    int defect_w = abs(defect.p1.x - defect.p2.x);
    int defect_h = abs(defect.p1.y - defect.p2.y);
    int judge_area = defect_w * defect_h;
    sprintf_alg("[defectInMask] defectInfo: judge_area=%d area=%d defect_w=%d, defect_h=%d", judge_area, defect.area, defect_w, defect_h);
    sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);

    // 截取缺陷区域
    cv::Mat img_mask = img;
    cv::Rect select = cv::Rect(defect.p1, defect.p2);
    cv::Mat ROI = img_mask(select);
    double sum = cv::sum(ROI)[0];
    sprintf_alg("[defectInMask] sum=%.3f", sum);
    //sprintf_alg("[defectInMask] sum=%.3f,judge_area * 0.7=%d", sum, (int)(judge_area * 0.7));

    // 在mask中
    if (sum > 0) {
        sprintf_alg("[defectInMask][import] defect Matched.");
        sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);
        result = true;
    }
    else {
        sprintf_alg("[defectInMask][import] have other defect!!!");
        sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);
    }
    sprintf_alg("[defectInMask][Out] result=%s", result ? "true" : "false");
    return result;
}

// 给缺陷分组
bool CReJudgeBack::getDefectsGroup(cv::Mat img_mask, vector<CDefect> v_defect) {
    sprintf_alg("");
    sprintf_alg("[getDefectsGroup][Enter]");
    // 遍历
    for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
        sprintf_alg("[getDefectsGroup]      defect_area=%d, defect_type = %d.", (*it).area, (*it).type);

        // 判断是否中间禁区
        bool matched = defectMatchBoxCenter(img_mask, *it);
        if (matched) {
            //sprintf_alg("[getDefectsGroup]       m_defectsInMask size : %d", m_defectsInCenter.size());
            continue;
        }

        // 判断在那个边角矩形组
        matched = defectMatchBox(img_mask, *it);
        if (matched) {
            //sprintf_alg("[getDefectsGroup]       m_defectsInMask size : %d", m_defectsInCenter.size());
            continue;
        }
        else {
            sprintf_alg("[getDefectsGroup][erroring]       match failed: defect_area=%d", (*it).area);
            m_defectsOthers.push_back(*it);
            sprintf_alg("[getDefectsGroup][erroring]       m_defectsOthers size : %d", m_defectsOthers.size());
        }
    }
    sprintf_alg("[getDefectsGroup][Outer]");
    return true;
}

// 匹配是不是在中间禁区
bool CReJudgeBack::defectMatchBoxCenter(cv::Mat img, CDefect defect) {
    sprintf_alg("[defectMatchBoxCenter][Enter] defect aera", defect.area);
    int result = false;

    // 创建缺陷mask
    cv::Mat img_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, cv::Scalar(1), -1, 4);

    for (auto it = m_imgCfgCenter.begin(); it != m_imgCfgCenter.end(); ++it) {
        //sprintf_alg("[defectMatchBox] m_imgCf: name=%s, arr_name=%s", (*it).name.c_str(), (*it).arr_name.c_str());
        string arr_name = (*it).arr_name;
        int cfg_area = (*it).area;
        int defect_w = abs(defect.p1.x - defect.p2.x);
        int defect_h = abs(defect.p1.y - defect.p2.y);
        int judge_area = defect_w * defect_h;
        //sprintf_alg("[defectMatchBox] defect_w=%d, defect_h=%d", defect_w, defect_h);
        cv::Rect select = cv::Rect((*it).p1, (*it).p2);
        cv::Mat ROI = img_mask(select);
        double sum = cv::sum(ROI)[0];
        sprintf_alg("[defectMatchBoxCenter] sum=%.3f,defect_x:p1.x=%d p2.x=%d config_x:x1=%d x2=%d", sum, defect.p1.x, defect.p2.x, (*it).p1.x, (*it).p1.x);
        sprintf_alg("[defectMatchBoxCenter] sum=%.3f,defect_x:p1.y=%d p2.y=%d config_y:y1=%d y2=%d", sum, defect.p1.y, defect.p2.y, (*it).p1.y, (*it).p1.y);
        // 有交集认为是这个的
        if (sum > 0) {
            m_defectsInCenter.push_back(defect);
            sprintf_alg("[defectMatchBoxCenter]       m_defectsInCenter center size: size=%d", m_defectsInCenter.size());
            result = true;
            break;
        }
    }
    sprintf_alg("[defectMatchBoxCenter][Outer] result=%s", result ? "true" : "false");
    return result;
}

// 判断在那个边角矩形组
bool CReJudgeBack::defectMatchBox(cv::Mat img, CDefect defect) {
    sprintf_alg("[defectMatchBox][Enter] defect aera", defect.area);
    int result = false;

    // 创建缺陷mask
    cv::Mat img_mask = cv::Mat::zeros(img.size(), CV_8UC1);
    rectangle(img_mask, { defect.p1, defect.p2 }, cv::Scalar(1), -1, 4);

    for (auto it = m_imgCfg.begin(); it != m_imgCfg.end(); ++it) {
        //sprintf_alg("[defectMatchBox] m_imgCf: name=%s, arr_name=%s", (*it).name.c_str(), (*it).arr_name.c_str());
        string arr_name = (*it).arr_name;
        int cfg_area = (*it).area;
        int defect_w = abs(defect.p1.x - defect.p2.x);
        int defect_h = abs(defect.p1.y - defect.p2.y);
        int judge_area = defect_w * defect_h;
        //sprintf_alg("[defectMatchBox] defect_w=%d, defect_h=%d", defect_w, defect_h);
        cv::Rect select = cv::Rect((*it).p1, (*it).p2);
        cv::Mat ROI = img_mask(select);
        double sum = cv::sum(ROI)[0];
        //sprintf_alg("[defectMatchBox] sum=%.3f,defect_x:p1.x=%d p2.x=%d config_x:x1=%d x2=%d", sum, defect.p1.x, defect.p2.x, x1, x2);
        // 有交集认为是这个的
        if (sum > 0) {
            m_DefectGroup[(*it).serial - 1].push_back(defect);
            sprintf_alg("[defectMatchBox]       m_DefectGroup size: serial=%d, size=%d", (*it).serial, m_DefectGroup[(*it).serial - 1].size());
            result = true;
        }
    }
    sprintf_alg("[defectMatchBox][Out] result=%s", result ? "true" : "false");
    return result;
}