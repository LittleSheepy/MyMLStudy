/*
2023年5月8日
*/
#pragma once
#include <opencv.hpp>
#include "CAlgBase.h"
#include "ReJudgeData.h"
class CReJudgeFront : public CAlgBase
{
private:
    void imgCfgInit();
    bool defectInMask(cv::Mat img, CDefect defect, int imgSerial);
    bool getPoint(cv::Mat img_gray, int imgSerial);
public:
    CReJudgeFront();
    bool Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect);
public:
    string					    m_className = "【正面二次复判】";
    map<int, vector<CDefect>>	m_DefectMatched;
    cv::Point                   m_Point;

    vector<CBox>			    m_img1Cfg;
    vector<CBox>			    m_img2Cfg;
    vector<CBox>			    m_img3Cfg;
    vector<CBox>			    m_img4Cfg;
    vector<vector<CBox>>	    m_imgCfg;
    map<int, vector<CDefect>>	m_CenterDefectMatched;
    map<int, vector<CDefect>>	m_BottomDefectMatched;
    map<string, int>		    m_brokenCnt;
    map<string, int>		    m_brokenCfg;
    map<string, vector<int>>	m_brokenCfg_vect;
};

