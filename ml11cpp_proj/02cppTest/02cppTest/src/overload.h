#pragma once


class MyClass
{
public:
	MyClass();
	~MyClass();

public:
	void myFunc(int para1);
	void myFunc(int para1, int para2=2);
private:

};


void overloadTest(void);