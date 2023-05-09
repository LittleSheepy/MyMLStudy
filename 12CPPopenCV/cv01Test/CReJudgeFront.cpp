/*
2023��5��8��
*/
#include "CReJudgeFront.h"

#define RJF_DEBUG 0

CReJudgeFront::CReJudgeFront() {
}

bool CReJudgeFront::getPoint(cv::Mat img_gray, int imgSerial) {
    vector<cv::Point> leftPointsList;
    vector<cv::Point> rightPointsList;
    vector<cv::Point> topPointsList;
    vector<cv::Point> bottomPointsList;
    // ��ȡ���ŵ� 20���� 
    for (int i = 500; i <= 1200; i += 50) {
        map<string, cv::Point> PointsTmp = getRowPoint(img_gray, i);     // 
        leftPointsList.push_back(PointsTmp["whiteleft"]);
        rightPointsList.push_back(PointsTmp["whiteright"]);
    }
    // ��ȡ���ŵ� 16���� 
    for (int i = 800; i <= 1800; i += 50) {
        map<string, cv::Point> PointsTmp = getColumnPoint(img_gray, i);     // 
        topPointsList.push_back(PointsTmp["whitetop"]);
        bottomPointsList.push_back(PointsTmp["whitebottom"]);
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
        m_Point = getIntersectionPoint(leftLine, topLine);
        break;
    case 1:
        fitLine(leftPointsList, leftLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(bottomPointsList, bottomLine, cv::DIST_L2, 0, 0.01, 0.01);
        m_Point = getIntersectionPoint(leftLine, bottomLine);
        break;
    case 2:
        fitLine(rightPointsList, rightLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(topPointsList, topLine, cv::DIST_L2, 0, 0.01, 0.01);
        m_Point = getIntersectionPoint(rightLine, topLine);
        break;
    case 3:
        fitLine(rightPointsList, rightLine, cv::DIST_L2, 0, 0.01, 0.01);
        fitLine(bottomPointsList, bottomLine, cv::DIST_L2, 0, 0.01, 0.01);
        m_Point = getIntersectionPoint(rightLine, bottomLine);
        break;
    default:
        break;
    }
    return true;
}

// ���� true��OK  false��NG
bool CReJudgeFront::Process(vector<cv::Mat> v_img, vector<vector<CDefect>> vv_defect) {
    sprintf_alg("[ReJudgeFront][Process][Begin]");
    // ���ù��̱���
    reset();
    m_DefectMatched.clear();
#ifdef RJF_DEBUG
    savePara(v_img, vv_defect, "./img_save/AI_para/ReJudgeFront/");
#endif // RJF_DEBUG
    bool result = true;
    m_Point = cv::Point(530, 30);

    vector<vector<CDefect>> vv_defect_others;
    // ����4��ͼƥ���ÿ�
    for (int i = 0; i < 4; i++) {
        cv::Mat img = v_img[i];
        getPoint(img, i);
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
                if ((*it).type == 11) {
                    bool matched = defectInMask(img, *it, i);
                    if (matched == false) {
                        // TODO �������ֱ�ӷ���false
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
        // ����ȱ��
        if (vv_defect_others[i].size() > 0) {
            result = false;
            sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d,have other broken defect!", i);
        }
        vector<CDefect> v_defectInMask = m_DefectMatched[i];
        vector<vector<CDefect>> vv_defectsGroup;
        // �ϲ������
        vv_defectsGroup = groupBBoxes(v_defectInMask);
        // ����ÿ�������
        int distence1 = 280;
        int distence2 = 880;
        for (vector<CDefect> v_defect : vv_defectsGroup) {
            vector<int> resultXYXY = getGroupBBoxesXYXY(v_defect);
            // �ж�Y
            int distence_y1 = abs(m_Point.y - resultXYXY[1]);
            int distence_y2 = abs(m_Point.y - resultXYXY[3]);
            // Y ���� NG
            if (((distence_y1 > distence1) && (distence_y1 < distence2)) || ((distence_y2 > distence1) && (distence_y2 < distence2))) {
                result = false;
                sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d, Y have defect!", i);
            }

            // �ж�X
            int distence_x1 = abs(m_Point.x - resultXYXY[0]);
            int distence_x2 = abs(m_Point.x - resultXYXY[2]);
            // X ���� NG
            if (((distence_x1 > distence1) && (distence_x1 < distence2)) || ((distence_x2 > distence1) && (distence_x2 < distence2))) {
                result = false;
                sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d, X have defect!", i);
            }

            // ��С
            int NG_LENGTH = 50;
            int defect_group_w = resultXYXY[2] - resultXYXY[0];
            int defect_group_h = resultXYXY[3] - resultXYXY[1];
            if ((defect_group_w > NG_LENGTH) || (defect_group_h > NG_LENGTH)) {
                result = false;
                sprintf_alg("[Process] ReJudgeFront is NG, img_num=%d,have Too Big defect!", i);
            }
        }
    }
    sprintf_alg("[ReJudgeFront][Process][End] result=%s", result ? "true" : "false");
    return result;
}

// ���� true����mask�� false�ǲ���mask��
bool CReJudgeFront::defectInMask(cv::Mat img, CDefect defect, int imgSerial) {
    sprintf_alg("[defectInMask][Enter] img serial=%d, defect aera", imgSerial, defect.area);
    int result = false;

    // ����ȱ����ӿ������WH
    int defect_w = abs(defect.p1.x - defect.p2.x);
    int defect_h = abs(defect.p1.y - defect.p2.y);
    int judge_area = defect_w * defect_h;
    sprintf_alg("[defectInMask] defectInfo: judge_area=%d area=%d defect_w=%d, defect_h=%d", judge_area, defect.area, defect_w, defect_h);
    sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);

    // ��ȡȱ������
    cv::Mat img_mask = img;
    cv::Rect select = cv::Rect(defect.p1, defect.p2);
    cv::Mat ROI = img_mask(select);
    double sum = cv::sum(ROI)[0];
    sprintf_alg("[defectInMask] sum=%.3f,judge_area * 0.7=%d", sum, (int)(judge_area * 0.7));

    // ��mask��
    if (sum > judge_area * 0.7) {
        m_DefectMatched[imgSerial].push_back(defect);
        sprintf_alg("[defectInMask]       m_DefectMatched %d size : %d", imgSerial, m_DefectMatched[imgSerial].size());
        result = true;
    } else {
        sprintf_alg("[defectInMask][import] img serial=%d have other defect!!!", imgSerial);
        sprintf_alg("[defectInMask] defectInfo: p1.x=%d p1.y=%d p2.x=%d p2.y=%d", defect.p1.x, defect.p1.y, defect.p2.x, defect.p2.y);
    }
    sprintf_alg("[defectInMask][Out] result=%s", result ? "true" : "false");
    return result;
}

