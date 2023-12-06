#define _CRT_SECURE_NO_WARNINGS
#include "pch.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "func.h"
#include <cstring>
using namespace std;
int hello()
{
    printf("hello world\n");
    return 0;
}

float func_type_base(float a, float b)
{
    printf("cpp: a = %f, b = %f \n", a, b);
    return a + b;
}

void func_arg_ptr(int* pInt, float* pFloat)
{
    cout << "cpp: pInt = " << pInt << endl;
    cout << "cpp: pFloat = " << pFloat << endl;
    cout << "cpp: *pInt = " << *pInt << endl;
    cout << "cpp: *pFloat = " << *pFloat << endl;
    ++*pInt;
    *pFloat = *pFloat + 0.5;
    cout << "cpp: *pInt = " << *pInt << endl;
    cout << "cpp: *pFloat = " << *pFloat << endl;
    printf("\n");
}

void func_arg_array(char* pChar, unsigned char* puStr, char* pszStr)
{
    cout << pChar << endl;
    cout << puStr << endl;
    for (int i = 0; i < 3; i++)
        cout << pChar[i];
    cout << endl;
    for (int i = 0; i < 3; i++)
        cout << puStr[i];
    cout << endl;

    pChar[0] = '2';
    pChar[1] = '1';
    puStr[0] = 'B';
    strcpy_s(pszStr,32, "Happay Children's day");
}