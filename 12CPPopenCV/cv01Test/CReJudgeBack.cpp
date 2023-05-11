/*
2023年5月11日
*/
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "CReJudgeBack.h"

#define RJB_DEBUG 0

#define PIX_H           18                        // 
#define MM_H            130
#define PIX_MM          (PIX_H/MM_H)                // 6.9
#define LEN2MM          int(PIX_MM*2)              // 14
#define LEN_PIX         580
#define W_PIX           100
#define EXPANSION       30

CReJudgeBack::CReJudgeBack() {

}

void CReJudgeBack::imgCfgInit() {
    m_imgCfg = {
        {"1", "tl", 1, m_Point_lt, cv::Point(m_Point_lt.x + LEN_PIX,m_Point_lt.y + W_PIX),LEN2MM},
        {"2", "lt", 2, m_Point_lt, cv::Point(m_Point_lt.x + W_PIX,m_Point_lt.y + LEN_PIX),LEN2MM},

        {"3", "bl", 3, m_Point_lb, cv::Point(m_Point_lb.x + LEN_PIX,m_Point_lb.y - W_PIX),LEN2MM},
        {"4", "lb", 4, m_Point_lb, cv::Point(m_Point_lb.x + W_PIX,m_Point_lb.y - LEN_PIX),LEN2MM},

        {"5", "tr", 5, m_Point_rt, cv::Point(m_Point_rt.x - LEN_PIX,m_Point_rt.y + W_PIX),LEN2MM},
        {"6", "rt", 6, m_Point_rt, cv::Point(m_Point_rt.x - W_PIX,m_Point_rt.y + LEN_PIX),LEN2MM},

        {"7", "br", 7, m_Point_rb, cv::Point(m_Point_rb.x - LEN_PIX,m_Point_rb.y - W_PIX),LEN2MM},
        {"8", "rb", 8, m_Point_rb, cv::Point(m_Point_rb.x - W_PIX,m_Point_rb.y - LEN_PIX),LEN2MM},
    };
    m_imgCfgCenter = {
        {"1", "tc", 1, cv::Point(m_Point_lt.x + LEN_PIX,m_Point_lt.y), cv::Point(m_Point_rt.x - LEN_PIX,m_Point_rt.y + W_PIX + 70),0},
        {"2", "bc", 2, cv::Point(m_Point_lb.x + LEN_PIX,m_Point_lb.y), cv::Point(m_Point_rb.x - LEN_PIX,m_Point_rb.y - W_PIX - 70),0},

        {"3", "lc", 3, cv::Point(m_Point_lt.x,m_Point_lt.y + LEN_PIX), cv::Point(m_Point_lb.x + W_PIX + 70,m_Point_lb.y - LEN_PIX),0},
        {"4", "rc", 4, cv::Point(m_Point_rt.x - W_PIX - 70,m_Point_rt.y + LEN_PIX), cv::Point(m_Point_rb.x,m_Point_rb.y - LEN_PIX),0}
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
    return true;
}

void test_showcfg(cv::Mat img_mask, vector<CBox> cfg, vector<CBox> cfg2) {
    cv::Mat img_save;
    img_save = cv::imread("D:/02dataset/01work/05nanjingLG/06ReJudgeBack/testSimple/img.jpg");
    for (auto it : cfg) {
        cv::rectangle(img_save, it.p1, it.p2, cv::Scalar(255, 255, 0), 2);
    }
    for (auto it : cfg2) {
        cv::rectangle(img_save, it.p1, it.p2, cv::Scalar(255, 0, 255), 2);
    }
    cv::imwrite("./img_save/CReJudgeBackCfg.jpg", img_save);
}

// 返回 true是OK  false是NG
bool CReJudgeBack::Process(cv::Mat img_mask, vector<CDefect> v_defect) {
    sprintf_alg("[ReJudgeBack][Process][Begin]");
    bool result = true;
    // 空图
    if(img_mask.empty()){
        sprintf_alg("[Process][import] img is empty.");
        result = true;
        sprintf_alg("[ReJudgeFront][Process][End] result=%s", result ? "true" : "false");
        return result;
    }
    // 空缺陷
    if (v_defect.size() == 0) {
        sprintf_alg("[Process][import] img have None defect.");
        result = true;
        sprintf_alg("[ReJudgeFront][Process][End] result=%s", result ? "true" : "false");
        return result;
    }
    else {
        sprintf_alg("[Process][import] img v_defect.size=%d", v_defect.size());
    }
    // 重置过程变量
    reset();
    m_DefectMatched.clear();

#ifdef RJB_DEBUG
    savePara(img_mask, v_defect, "./img_save/AI_para/ReJudgeBack/");
#endif // RJB_DEBUG
    getPoint(img_mask);

    imgCfgInit();
    test_showcfg(img_mask, m_imgCfgCenter, m_imgCfg);
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

    // 复判缺陷 边中间有缺陷
    if (m_defectsInCenter.size() > 0) {
        result = false;
        sprintf_alg("[Process] ReJudgeFront is NG,Center have broken defect!");
        // TODO 上线可以在这里直接返回false
        //sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
        //return result;
    }
    
    // 判断边角矩形框缺陷个数超限制
    for (const auto& pair : m_DefectGroup) {
        int key = pair.first;
        const vector<CDefect>& v_defect1 = pair.second;
        vector<vector<CDefect>> vv_defect1 = groupBBoxes(v_defect1);

        int defect_cnt = 0;
        for (vector<CDefect> v_defect2 : vv_defect1) {
            vector<int> resultWH = getGroupBBoxesWH(v_defect2);
            int defect_length = std::max(resultWH[0], resultWH[1]);
            if (defect_length > LEN2MM) {
                defect_cnt++;
            }
        }
        if (defect_cnt >=2) {
            sprintf_alg("[Process] ReJudgeFront is NG,key=%d, Center have broken defect!num is too big", key);
            result = false;
            // TODO 上线可以在这里直接返回false
            //sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
            //break;
        }
    }

    sprintf_alg("[ReJudgeBack][Process][End] result=%s", result ? "true" : "false");
    return result;
}

bool CReJudgeBack::getDefectsInMask(cv::Mat img_mask, vector<CDefect> v_defect) {
    // 二值化
    cv::Mat img;
    cv::threshold(img_mask, img, 50, 255, cv::THRESH_BINARY);
    // 遍历
    for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
        //sprintf_alg("[Process]      defect_area=%d, defect_type = %d.", (*it).area, (*it).type);
        if ((*it).area > 0) {
            if ((*it).type == 11) {
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
            else {
                sprintf_alg("[getDefectsInMask][warring] type=%d name=%s", (*it).type, (*it).name.c_str());
            }
        }
        else {
            sprintf_alg("[getDefectsInMask][warring] area=%d", (*it).area);
        }
    }
    return true;
}
// 返回 true是在mask中 false是不在mask中
bool CReJudgeBack::defectInMask(cv::Mat img, CDefect defect) {
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
    sprintf_alg("[getDefectsGroup][Enter]");
    // 遍历
    for (auto it = v_defect.begin(); it != v_defect.end(); ++it) {
        if ((*it).area <= 0) {
            sprintf_alg("[getDefectsGroup][warring]     defect_area is 0, area=%d", (*it).area);
            continue;
        }
        if ((*it).type != 11) {
            sprintf_alg("[getDefectsGroup][warring]     type=%d name=%s", (*it).type, (*it).name.c_str());
            continue;
        }
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
        //sprintf_alg("[defectMatchBoxCenter] sum=%.3f,defect_x:p1.x=%d p2.x=%d config_x:x1=%d x2=%d", sum, defect.p1.x, defect.p2.x, x1, x2);
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
            sprintf_alg("[defectMatchBox]       m_DefectGroup size: serial=%d, size=%d", (*it).serial, m_DefectGroup[(*it).serial].size());
            result = true;
        }
    }
    sprintf_alg("[defectMatchBox][Out] result=%s", result ? "true" : "false");
    return result;
}