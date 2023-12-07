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

void func_arg_struct(Rect rect, Rect* pRect)
{
    cout << ("cpp: value===========================") << endl;
    cout << "cpp: rect index,info: " << rect.index << "," << rect.info << endl;
    cout << "cpp: prect index,info: " << pRect->index << "," << pRect->info << endl;
    rect.index = rect.index + 10;
    pRect->index = pRect->index + 10;
    cout << "cpp: index,info: " << rect.index << "," << rect.info << endl;
    cout << "cpp: prect index,info: " << pRect->index << "," << pRect->info << endl;
}

void func_arg_struct_array(Rect* pRectArray)
{
    for (int i = 0; i < 5; i++)
        cout << "cpp: prect index,info: " << pRectArray[i].index << "," << pRectArray[i].info << endl;
   
    pRectArray[0].index += 10;

    for (int i = 0; i < 5; i++)
        cout << "cpp: prect index,info: " << pRectArray[i].index << "," << pRectArray[i].info << endl;
}

Rect* func_res_struct_array(int* pArrayNum)
{
    int num = 5;
    *pArrayNum = num;
    cout << "cpp: pArrayNum:" << *pArrayNum << endl;
    Rect* pArray = (Rect*)malloc(num * sizeof(Rect));
    for (int i = 0; i < num; i++)
    {
        pArray[i].index = i;
        sprintf_s(pArray[i].info, "%s_%d", "hello", i);
    }

    return pArray;
}

void freeRect(Rect* pRect)
{
    free(pRect);
}

void Print_Company(CompanyStruct* com)
{
    cout << "cpp: id, name, count:  " <<com->com_id << "," << com->name << "," << com->count << endl;
    for (int i = 0; i < com->count; i++)
    {
        cout << "cpp:  " << com->users[i].user_id << "," << com->users[i].name << endl;
    }

    memset(com, 0, sizeof(CompanyStruct));

    com->com_id = 1001;
    strncpy_s(com->name, "esunny_cpp", sizeof(com->name));
    com->count = 3;
    for (int i = 0; i < com->count; i++)
    {
        com->users[i].user_id = i;
        char key[21] = { 0 };
        snprintf(key, sizeof(key), "user_cpp_%d", i);
        strncpy_s(com->users[i].name, key, sizeof(com->users[i].name));
    }
}


