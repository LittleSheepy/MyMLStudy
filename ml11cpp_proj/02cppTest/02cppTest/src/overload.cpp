#include <cstdio>

#include "overload.h"

MyClass::MyClass()
{
}

MyClass::~MyClass()
{
}

void MyClass::myFunc(int para1)
{
    printf("myFunc one para 1\n");
}

void MyClass::myFunc(int para1, int para2)
{
    printf("myFunc one para 2\n");
}
void overloadTest(void)
{
    MyClass my_class = MyClass();
    int i = 1;
    //my_class.myFunc(i);
    my_class.myFunc(1, 2);

}