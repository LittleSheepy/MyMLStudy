// pch.cpp: 与预编译标头对应的源文件

#include "pch.h"

// 当使用预编译的头时，需要使用此源文件，编译才能成功。

#ifdef _DEBUG

//#pragma comment( lib, "..\\lib\\opencv_world455d.lib")
//#pragma comment( lib, "..\\lib\\libJsond.lib")
//#pragma comment( lib, "..\\lib\\KDLib_IOd.lib")
//#pragma comment( lib, "..\\lib\\KDLib_Barcoded.lib")
//#pragma comment( lib, "..\\lib\\KDLogd.lib")

#else

//#pragma comment( lib, "..\\lib\\opencv_world455.lib")
//#pragma comment( lib, "..\\lib\\libJson.lib")
//#pragma comment( lib, "..\\lib\\KDLib_IO.lib")
//#pragma comment( lib, "..\\lib\\KDLib_Barcode.lib")
//#pragma comment( lib, "..\\lib\\KDLog.lib")

#endif
