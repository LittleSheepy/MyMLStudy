/*
2023年5月19日
*/
#pragma once
#include <opencv.hpp>
#include "CAlgBase.h"

struct CfgRJBStruct
{
    int len_pix = 580;
    int w_pix = 120;
    int expansion = 30;
    int len_hole = 30;
    int expansion_corner = 0;

};
class CReJudgeBack : public CAlgBase
{
private:
    bool filterDefects(vector<CDefect>& v_defect, vector<CDefect>& v_defect_need, vector<CDefect>& v_defect_others);
    bool defectInMask(cv::Mat img, CDefect defect);
    bool getDefectsInMask(cv::Mat img_mask, vector<CDefect> v_defect);
    bool getDefectsInMask2(cv::Mat img, vector<CDefect>& v_defect, vector<CDefect>& v_defect_matched, vector<CDefect>& v_defect_others);
    bool getPoint(cv::Mat img_gray);
    bool getDefectsGroup(cv::Mat img_mask, vector<CDefect> v_defect);
    bool defectMatchBox(cv::Mat img, CDefect defect);                   // 匹配在边角矩形组
    bool defectMatchBoxCenter(cv::Mat img, CDefect defect);             // 匹配是不是在中间禁区
public:
    bool loadCfg();
    void imgCfgInit();
    CReJudgeBack();
    bool Process(cv::Mat img_mask, vector<CDefect> v_defect);
public:
    string					        m_className = "【底面二次复判】";
    vector<CDefect>	                m_DefectMatched;
    vector<CDefect>	                m_defectsInEdge;
    vector<CDefect>	                m_defectsInLocHole;
    vector<CDefect>	                m_defectsOthers;
    vector<CDefect>	                m_defectsInCenter;
    map<int, vector<CDefect>>	    m_DefectGroup;
    cv::Point                       m_Point_lt;
    cv::Point                       m_Point_lb;
    cv::Point                       m_Point_rt;
    cv::Point                       m_Point_rb;
    vector<CBox>			        m_imgCfg;
    vector<CBox>			        m_imgCfgCenter;
    CfgRJBStruct                    m_cfgRJB;
    int                             m_len5mm;
    bool                            m_otherTypeDefect;
};

