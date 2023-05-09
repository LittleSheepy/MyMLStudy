/*
2023年5月8日
*/
#pragma once
#include <opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "CAlgBase.h"
using namespace std;

// 瑕疵结构体
struct CDefect
{
    cv::Point   p1;	        // 左上点
    cv::Point   p2;	        // 右下点
    int         area;	    // 缺陷面积
    int	        type;	    // 缺陷类型 11破损、12毛边
    string	    name;	    // 缺陷类型 11破损、12毛边
    double      realArea;   // 真实面积
    CDefect() {}
    CDefect(cv::Point p1, cv::Point p2, int area = 0, int type = 11, string name = "null", double realArea = 0)
        :p1(p1), p2(p2), area(area), type(type), name(name), realArea(realArea) {}

    bool operator==(const CDefect& other) const {
        return p1.x == other.p1.x && p1.y == other.p1.y && p2.x == other.p2.x && p2.y == other.p2.y;
    }
    bool overlap(const CDefect& other) {
        return (p1.x <= other.p2.x && p2.x >= other.p1.x && p1.y <= other.p2.y && p2.y >= other.p1.y);
    }
};

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
class CBoxArray {
public:
    vector<CBox> v_obj;
    int state;
};
