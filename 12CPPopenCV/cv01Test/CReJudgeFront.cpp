/*
2023年5月19日
*/
#include "pch.h"
#include <fstream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem> // C++14标准引入的文件系统库
namespace fs = std::experimental::filesystem;

#include "CReJudgeFront.h"

#define RJF_DEBUG 0

CReJudgeFront::CReJudgeFront() {
    loadCfg();
}

bool CReJudgeFront::loadCfg() {
    if (!fs::exists("Config/ReJudgeFront.cfg")) {
        sprintf_alg("[CReJudgeFront][error] cfg file not exist!!!");
        return false;
    }
    std::ifstream ifs("Config/ReJudgeFront.cfg");
    Json::Reader reader;
    Json::Value m_imgCfgJson;
    reader.parse(ifs, m_imgCfgJson);

    m_v_cfg.clear();
    for (const auto& camera : m_imgCfgJson) {
        CfgStruct cameraCfg;
        cameraCfg.distence1 = camera["distence1"].asInt();
        cameraCfg.distence2 = camera["distence2"].asInt();
        m_v_cfg.push_back(cameraCfg);
    }
    return true;
}
bool CReJudgeFront::getPoint(cv::Mat img_gray, int imgSerial) {
    vector<cv::Point> leftPointsList;
    vector<cv::Point> rightPointsList;
    vector<cv::Point> topPointsList;
    vector<cv::Point> bottomPointsList;
    // 获取竖着的 20个点 
    for (int i = 500; i <= 1200; i += 50) {
        map<string, cv::Point> PointsTmp = getRowPoint(img_gray, i);     // 
        leftPointsList.push_back(PointsTmp["whiteleft"]);
        rightPointsList.push_back(PointsTmp["whiteright"]);
        //sprintf_alg("[Process]     >>>>>> img_num=%d, whiteleft=%d, %d!", imgSerial, PointsTmp["whiteleft"].x, PointsTmp["whiteleft"].y);
        //sprintf_alg("[Process]     >>>>>> img_num=%d, whiteright=%d, %d!", imgSerial, PointsTmp["whiteright"].x, PointsTmp["whiteright"].y);
    }
    // 获取横着的 16个点 
    for (int i = 800; i <= 1800; i += 50) {
        map<string, cv::Point> PointsTmp = getColumnPoint(img_gray, i);     // 
        topPointsList.push_back(PointsTmp["whitetop"]);
        bottomPointsList.push_back(PointsTmp["whitebottom"]);
        //sprintf_alg("[Process]     >>>>>> img_num=%d, whitetop=%d, %d!", imgSerial, PointsTmp["whitetop"].x, PointsTmp["whitetop"].y);
        //sprintf_alg("[Process]     >>>>>> img_num=%d, whitebottom=%d, %d!", imgSerial, PointsTmp["whitebottom"].x, PointsTmp["whitebottom"].y);
    }
    cv::Vec4f leftLine;
    cv::Vec4f rightLine;
    cv::Vec4f topLine;
    cv::Vec4f bottomLine;
    switch (imgSerial)
    {
    case 0:
        fitLine(leftPointsList, leftLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(topPointsList, topLine, cv::DIST_L2, 0, 0.01, 0.01);
        //sprintf_alg("[Process]>>>>>> img_num=%d, leftLine=%f, %f, %f, %f!", imgSerial, leftLine[0], leftLine[1], leftLine[2], leftLine[3]);
        //sprintf_alg("[Process]>>>>>> img_num=%d, topLine=%f, %f, %f, %f!", imgSerial, topLine[0], topLine[1], topLine[2], topLine[3]);
        m_Point = getIntersectionPoint(leftLine, topLine);
        break;
    case 1:
        fitLine(leftPointsList, leftLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(bottomPointsList, bottomLine, cv::DIST_L2, 0, 0.01, 0.01);
        //sprintf_alg("[Process]>>>>>> img_num=%d, leftLine=%f, %f, %f, %f!", imgSerial, leftLine[0], leftLine[1], leftLine[2], leftLine[3]);
        //sprintf_alg("[Process]>>>>>> img_num=%d, bottomLine=%f, %f, %f, %f!", imgSerial, bottomLine[0], bottomLine[1], bottomLine[2], bottomLine[3]);
        m_Point = getIntersectionPoint(leftLine, bottomLine);
        break;
    case 2:
        fitLine(rightPointsList, rightLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(topPointsList, topLine, cv::DIST_L2, 0, 0.01, 0.01);
        //sprintf_alg("[Process]>>>>>> img_num=%d, rightLine=%f, %f, %f, %f!", imgSerial, rightLine[0], rightLine[1], rightLine[2], rightLine[3]);
        //sprintf_alg("[Process]>>>>>> img_num=%d, topLine=%f, %f, %f, %f!", imgSerial, topLine[0], topLine[1], topLine[2], topLine[3]);
        m_Point = getIntersectionPoint(rightLine, topLine);
        break;
    case 3:
        fitLine(rightPointsList, rightLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(bottomPointsList, bottomLine, cv::DIST_L2, 0, 0.01, 0.01);
        //sprintf_alg("[Process]>>>>>> img_num=%d, rightLine=%f, %f, %f, %f!", imgSerial, rightLine[0], rightLine[1], rightLine[2], rightLine[3]);
        //sprintf_alg("[Process]>>>>>> img_num=%d, bottomLine=%f, %f, %f, %f!", imgSerial, bottomLine[0], bottomLine[1], bottomLine[2], bottomLine[3]);
        m_Point = getIntersectionPoint(rightLine, bottomLine);
        break;
    default:
        break;
    }
    return true;
}

// 返回 true是OK  false是NG
bool CReJudgeFront::Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect) {
    sprintf_alg("[ReJudgeFront][Process][Begin]");
    sprintf_alg("[Process] img channels is %d.", v_img[0].channels());
    // 判断参数有效
    if (v_img.size() != 4) {
        sprintf_alg("\n[ReJudgeFront][error] para error!!! v_img is not 4!!!, v_img size=%d.\n", v_img.size());
        sprintf_alg("[ReJudgeFront][End] result=false");
        return false;
    }
    if (vv_defect.size() != 4) {
        sprintf_alg("\n[ReJudgeFront][error] para error!!! vv_defect is not 4!!!, vv_defect size=%d.\n", vv_defect.size());
        sprintf_alg("[ReJudgeFront][End] result=false");
        return false;
    }
    // 重置过程变量
    reset();
    m_DefectMatched.clear();
    m_otherTypeDefect = false;
#ifdef RJF_DEBUG
    savePara(v_img, vv_defect, "./img_save/AI_para/ReJudgeFront/");
#endif // RJF_DEBUG
    bool result = true;
    m_Point = cv::Point(530, 30);

    vector<vector<CDefect>> vv_defect_others;
    // 遍历4个图匹配置框
    for (int i = 0; i < 4; i++) {
        cv::Mat img;
        if (v_img[i].channels() != 1) {
            cv::Mat grayImage;
            cv::cvtColor(v_img[i], img, cv::COLOR_BGR2GRAY);
        }
        else {
            img = v_img[i];
        }
        getPoint(img, i);
        sprintf_alg("[Process]>>>>>> img_num=%d, m_Point=%d, %d!", i, m_Point.x, m_Point.y);
        vector<CDefect> v_defect = vv_defect[i];
        vv_defect_others.push_back({});
        if (v_defect.size() == 0) {
            sprintf_alg("[Process][import] img %d have None defect.", i);
            continue;
        }
        else {
            sprintf_alg("[Process][import] img %d v_defect.size=%d", i, v_defect.size());
        }
        for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
            //sprintf_alg("[Process]      defect_area=%d, defect_type = %d.", (*it).area, (*it).type);
            if ((*it).area > 0) {
                if ((*it).type == 10) {
                    bool matched = defectInMask(img, *it, i);
                    if (matched == false) {
                        // TODO 这里可以直接返回false
                        // result = false;
                        // break;
                        sprintf_alg("[Process] match failed: defect_area=%d", (*it).area);
                        vv_defect_others[i].push_back(*it);
                    }
                }
                else {
                    sprintf_alg("[Process][warring]type is not 10. type=%d name=%s", (*it).type, (*it).name.c_str());
                    m_otherTypeDefect = true;
                }
            }
            else {
                sprintf_alg("[Process][warring] area=%d", (*it).area);
            }
        }
        //sprintf_alg("[Process]      v_defect_others size = %d", vv_defect_others[i].size());
        // 复判缺陷
        if (vv_defect_others[i].size() > 0) {
            result = false;
            sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d,have other broken defect!", i);
        }
        vector<CDefect> v_defectInMask = m_DefectMatched[i];
        vector<vector<CDefect>> vv_defectsGroup;
        // 合并破损框
        vv_defectsGroup = groupBBoxesByType(v_defectInMask, 0);
        // 遍历每组破损框
        int camera = i % 2;
        int distence1 = m_v_cfg[camera].distence1;
        int distence2 = m_v_cfg[camera].distence2;
        for (vector<CDefect> v_defect : vv_defectsGroup) {
            vector<int> resultXYXY = getGroupBBoxesXYXY(v_defect);
            sprintf_alg("[Process] img_num=%d, resultXYXY: %d %d %d %d!", i, resultXYXY[0], resultXYXY[1], resultXYXY[2], resultXYXY[3]);
            // 判断Y
            int distence_y1 = abs(m_Point.y - resultXYXY[1]);
            int distence_y2 = abs(m_Point.y - resultXYXY[3]);
            sprintf_alg("[Process]  img_num=%d, distence_y1=%d, distence_y2=%d!", i, distence_y1, distence_y2);
            // Y 方向 NG
            if (((distence_y1 > distence1) && (distence_y1 < distence2)) || ((distence_y2 > distence1) && (distence_y2 < distence2))) {
                result = false;
                sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d, have defect !!!", i);
                continue;
            }

            // 判断X
            int distence_x1 = abs(m_Point.x - resultXYXY[0]);
            int distence_x2 = abs(m_Point.x - resultXYXY[2]);
            sprintf_alg("[Process] img_num=%d, distence_x1=%d, distence_x2=%d!", i, distence_x1, distence_x2);
            // X 方向 NG
            if (((distence_x1 > distence1) && (distence_x1 < distence2)) || ((distence_x2 > distence1) && (distence_x2 < distence2))) {
                result = false;
                sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d, have defect!", i);
                continue;
            }

            // 大小
            int NG_LENGTH = 14;
            int defect_group_w = resultXYXY[2] - resultXYXY[0];
            int defect_group_h = resultXYXY[3] - resultXYXY[1];
            sprintf_alg("[Process] img_num=%d, defect_group_w=%d, defect_group_h=%d!", i, defect_group_w, defect_group_h);
            if ((defect_group_w > NG_LENGTH) || (defect_group_h > NG_LENGTH)) {
                result = false;
                sprintf_alg("[Process] This defect ReJudgeFront is NG, img_num=%d,have Too Big defect!", i);
            }
            else {
                sprintf_alg("[Process] This defect ReJudgeFront is OK, img_num=%d,have Too Small defect!", i);
            }
        }
    }

    if (m_otherTypeDefect) {
        sprintf_alg("[ReJudgeFront][Process][error] Have other defect!!!");
        result = false;
    }
    sprintf_alg("[ReJudgeFront][Process][End] result=%s", result ? "true" : "false");
    return result;
}

// 返回 true是在mask中 false是不在mask中
bool CReJudgeFront::defectInMask(cv::Mat img, CDefect defect, int imgSerial) {
    sprintf_alg("[defectInMask][Enter] img serial=%d, defect aera", imgSerial, defect.area);
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
    sprintf_alg("[defectInMask] sum=%.3f,judge_area * 0.7=%d", sum, (int)(judge_area * 0.7));

    // 在mask中
    if (sum > judge_area * 0.7) {
        m_DefectMatched[imgSerial].push_back(defect);
        sprintf_alg("[defectInMask]       m_DefectMatched %d size : %d", imgSerial, m_DefectMatched[imgSerial].size());
        result = true;
    }
    else {
        sprintf_alg("[defectInMask][import] img serial=%d have other defect!!!", imgSerial);
        sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);
    }
    sprintf_alg("[defectInMask][Out] result=%s", result ? "true" : "false");
    return result;
}

