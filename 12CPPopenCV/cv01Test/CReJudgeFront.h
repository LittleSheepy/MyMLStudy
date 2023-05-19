/*
2023年5月19日
*/
#pragma once
#include <opencv.hpp>
#include "CAlgBase.h"
#include "ReJudgeData.h"
struct CfgStruct
{
    int distence1 = 280;
    int distence2 = 880;

};
class CReJudgeFront : public CAlgBase
{
private:
    bool defectInMask(cv::Mat img, CDefect defect, int imgSerial);
    bool getPoint(cv::Mat img_gray, int imgSerial);
public:
    CReJudgeFront();
    bool loadCfg();
    bool Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect);
public:
    string					    m_className = "【正面二次复判】";
    map<int, vector<CDefect>>	m_DefectMatched;
    cv::Point                   m_Point;
    std::vector<CfgStruct>      m_v_cfg;        // 0:camera0  1:camera1
    bool                        m_otherTypeDefect;
};

