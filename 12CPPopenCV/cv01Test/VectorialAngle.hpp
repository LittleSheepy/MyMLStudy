#pragma once

//计算两个向量的夹角
#include<cmath>
template <typename _T>
_T dot(_T x1, _T y1, _T x2, _T y2) {
    return x1 * x2 + y1 * y2;
}

template <typename _T>
_T cos_angle(_T x1, _T y1, _T x2, _T y2) {
    _T v1 = sqrt(x1 * x1 + y1 * y1);
    _T v2 = sqrt(x2 * x2 + y2 * y2);
    return dot<_T>(x1, y1, x2, y2) / (v1 * v2);
}

template <typename _T>
_T angle(_T x1, _T y1, _T x2, _T y2) {
    _T cos = cos_angle<_T>(x1, y1, x2, y2);
    return acos(cos) * 180 / 3.1415926;
}

//template <typename _T>
//_T* intersect(_T x1, _T y1, _T x2, _T y2, _T x3, _T y3, _T x4, _T y4) {
//    _T a1 = (y2 - y1) / (x2 - x1);
//    _T b1 = y1 - a1 * x1;
//    _T a2 = (y4 - y3) / (x4 - x3);
//    _T b2 = y3 - a2 * x3;
//    _T x = (b2 - b1) / (a1 - a2);
//    _T y = a1 * x + b1;
//    return [x, y];
//}