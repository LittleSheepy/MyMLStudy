#define _CRT_SECURE_NO_WARNINGS
#pragma once
#include "pch.h"


//½á¹¹Ìå
typedef struct _rect
{
    int index;
    char info[16];
} Rect;

struct UserStruct
{
    long user_id;
    char name[21];
};

struct CompanyStruct
{
    long com_id;
    char name[21];
    UserStruct users[100];
    int count;
};

#ifdef __cplusplus
extern "C" {
#endif

    DLL_API int hello();
    DLL_API float func_type_base(float a, float b);
    DLL_API void func_arg_ptr(int* pInt, float* pFloat);
    DLL_API void func_arg_array(char* pChar, unsigned char* puStr, char* pszStr);
    DLL_API void func_arg_struct(Rect rect, Rect* pRect);
    DLL_API void func_arg_struct_array(Rect* pRectArray);
    DLL_API Rect* func_res_struct_array(int* pArrayNum);
    DLL_API void freeRect(Rect* pRect);
    DLL_API void Print_Company(CompanyStruct* com);

#ifdef __cplusplus
}
#endif