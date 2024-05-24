#pragma once

#include<stdio.h>
#include<stdlib.h>

#define MAX 50

struct SeqString
{
    int MAXUM;
    int n;
    char* c;
};
typedef struct SeqString* PSeqString;
//创建空字符串
PSeqString createNullStr_seq(int m);

//判断串是否为空串
int isNullList(PSeqString p);

//返回串的长度
int length(PSeqString p);
//返回将串s1和串s2拼接在一起构成的新串
PSeqString concat(PSeqString s1, PSeqString s2);
//在串s中,求从串的第i个字符开始连续j个字符所组成的字串
PSeqString subStr(PSeqString s, int i, int j);
//如果串s2是s1的字串,则可求串s2在s1中第一次出现的位置
//带回溯的算法  非KMP   KMP算法待补充
int index1(PSeqString s1, PSeqString s2);
//判断是否为回文字符串
int isPalindromeStr(PSeqString p);

