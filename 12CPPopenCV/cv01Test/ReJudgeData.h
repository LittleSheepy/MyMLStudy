/*
2023��5��8��
*/
#pragma once
#include <opencv.hpp>
#include <vector>
#include <string>
#include <map>
#include "CAlgBase.h"
using namespace std;

// 覴ýṹ��
struct CDefect
{
    cv::Point   p1;	        // ���ϵ�
    cv::Point   p2;	        // ���µ�
    int         area;	    // ȱ�����
    int	        type;	    // ȱ������ 11����12ë��
    string	    name;	    // ȱ������ 11����12ë��
    double      realArea;   // ��ʵ���
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
    int serial;             // ���
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
