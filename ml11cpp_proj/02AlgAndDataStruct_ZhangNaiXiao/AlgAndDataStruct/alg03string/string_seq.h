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
//�������ַ���
PSeqString createNullStr_seq(int m);

//�жϴ��Ƿ�Ϊ�մ�
int isNullList(PSeqString p);

//���ش��ĳ���
int length(PSeqString p);
//���ؽ���s1�ʹ�s2ƴ����һ�𹹳ɵ��´�
PSeqString concat(PSeqString s1, PSeqString s2);
//�ڴ�s��,��Ӵ��ĵ�i���ַ���ʼ����j���ַ�����ɵ��ִ�
PSeqString subStr(PSeqString s, int i, int j);
//�����s2��s1���ִ�,�����s2��s1�е�һ�γ��ֵ�λ��
//�����ݵ��㷨  ��KMP   KMP�㷨������
int index1(PSeqString s1, PSeqString s2);
//�ж��Ƿ�Ϊ�����ַ���
int isPalindromeStr(PSeqString p);

