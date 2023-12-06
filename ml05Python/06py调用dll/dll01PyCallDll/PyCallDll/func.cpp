#include "pch.h"
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include "func.h"

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