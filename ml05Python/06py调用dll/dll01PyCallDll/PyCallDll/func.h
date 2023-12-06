#define _CRT_SECURE_NO_WARNINGS
#pragma once
#include "pch.h"
#ifdef __cplusplus
extern "C" {
#endif

    DLL_API int hello();
    DLL_API float func_type_base(float a, float b);
    DLL_API void func_arg_ptr(int* pInt, float* pFloat);
    DLL_API void func_arg_array(char* pChar, unsigned char* puStr, char* pszStr);

#ifdef __cplusplus
}
#endif