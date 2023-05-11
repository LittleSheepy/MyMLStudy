/*
2023年5月9日
*/
#pragma once
#include <opencv.hpp>
#include <core.hpp>
#include <imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "CAlgBase.h"
class CReJudgeBack : public CAlgBase
{
private:
    bool defectInMask(cv::Mat img, CDefect defect);
    bool getDefectsInMask(cv::Mat img_mask, vector<CDefect> v_defect);
    bool getPoint(cv::Mat img_gray);
    bool getDefectsGroup(cv::Mat img_mask, vector<CDefect> v_defect);
    bool defectMatchBox(cv::Mat img, CDefect defect);                   // 匹配在边角矩形组
    bool defectMatchBoxCenter(cv::Mat img, CDefect defect);             // 匹配是不是在中间禁区
public:
    void imgCfgInit();
    CReJudgeBack();
    bool Process(cv::Mat img_mask, vector<CDefect> v_defect);
public:
    string					        m_className = "【底面二次复判】";
    vector<CDefect>	                m_DefectMatched;
    vector<CDefect>	                m_defectsInMask;
    vector<CDefect>	                m_defectsOthers;
    vector<CDefect>	                m_defectsInCenter;

    map<int, vector<CDefect>>	    m_DefectGroup;
    cv::Point                       m_Point_lt;
    cv::Point                       m_Point_lb;
    cv::Point                       m_Point_rt;
    cv::Point                       m_Point_rb;
    vector<CBox>			        m_imgCfg;
    vector<CBox>			        m_imgCfgCenter;

};

