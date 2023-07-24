
#include "pch.h"
#include "framework.h"
//#include "TestDlg.h"
#include "TestDlgDlg.h"
#include "afxdialogex.h"

#include <opencv.hpp>
#include <io.h>
#include <direct.h>
//#include "CAlgo.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <io.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

//#include "..\KDLib_Barcode\Interface.h"
#include "CAlgo_KDWater2.h"

#include <vector>
#include <string>

using namespace std;
using namespace cv;



void _ScanFiles_NoPath(std::string path, std::vector<std::string>& vecfiles)
{
	//文件句柄  
	__int64   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;

	std::string affix = "\\*.*";
	if (path.size() <= 0)
	{
		affix = "*.*";
	}


	if ((hFile = _findfirst(p.assign(path).append(affix).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{   // 子文件夹
				if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))
				{  // 非. 和 .. ,即为子目录
					continue;

// 					if (path.size() <= 0)
// 					{
// 						_ScanFiles(fileinfo.name, vecfiles);
// 					}
// 					else
// 					{
// 						_ScanFiles(path + "\\" + fileinfo.name, vecfiles);
// 					}
				}
			}
			else
			{   // 文件
				std::string name = fileinfo.name;
				size_t pos0 = name.find(".jpg");
				size_t pos1 = name.find(".png");
				size_t pos2 = name.find(".bmp");
				if ((pos0 >= 0) || (pos1 >= 0) || (pos2 >= 0))
				{
					vecfiles.push_back(fileinfo.name);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}

// path, , 仅 图像文件
void _ScanFiles(std::string path, std::vector<std::string>& vecfiles)
{
	//文件句柄  
	__int64   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;

	std::string affix = "\\*.*";
	if (path.size() <= 0)
	{
		affix = "*.*";
	}


	if ((hFile = _findfirst(p.assign(path).append(affix).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{   // 子文件夹
				if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))
				{  // 非. 和 .. ,即为子目录
					if (path.size() <= 0)
					{
						_ScanFiles(fileinfo.name, vecfiles);
					}
					else
					{
						_ScanFiles(path + "\\" + fileinfo.name, vecfiles);
					}
				}
			}
			else
			{   // 文件
				std::string name = fileinfo.name;
				size_t pos0 = name.find(".jpg");
				size_t pos1 = name.find(".png");
				size_t pos2 = name.find(".bmp");
				if ((pos0 >= 0) || (pos1 >= 0) || (pos2 >= 0))
//				if ( pos2 >= 0 )
				{
					if (path.size() <= 0)
					{
						vecfiles.push_back(fileinfo.name);
					}
					else
					{
						vecfiles.push_back(path + "\\" + fileinfo.name);
					}
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}


// path, , 仅 Jpg图像文件
void _ScanFiles_Jpg(std::string path, std::vector<std::string>& vecfiles)
{
	//文件句柄  
	__int64   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;

	std::string affix = "\\*.*";
	if (path.size() <= 0)
	{
		affix = "*.*";
	}

	if ((hFile = _findfirst(p.assign(path).append(affix).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{   // 子文件夹
				continue;

				if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))
				{  // 非. 和 .. ,即为子目录
					//files.push_back(fileinfo.name);

					if (path.size() <= 0)
					{
						_ScanFiles(fileinfo.name, vecfiles);
					}
					else
					{
						_ScanFiles(path + "\\" + fileinfo.name, vecfiles);
					}
				}
			}
			else
			{  // 文件

				std::string name = fileinfo.name;
				size_t pos0 = name.find(".jpg");
				//int pos2 = name.find(".bmp");
				if ( pos0 >= 0 )
				{
					if (path.size() <= 0)
					{
						vecfiles.push_back(fileinfo.name);
					}
					else
					{
						vecfiles.push_back(path + "\\" + fileinfo.name);
					}
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}


#if 0

#include "CAlgo_KDWater.h"

void _Test_Water()
{
	CAlgo_KDWater2  water;
	water.LoadTemplate();
	
	std::vector<_TKDWater_Defect2>  vecDefect;
	
 	cv::Mat img = cv::imread( "D:\\1\\0.bmp", cv::IMREAD_GRAYSCALE);
 	// water.FindDefect(vecDefect, img);

	std::vector<std::string>  vecFiles;
	// _ScanFiles( "D:\\LGCBTray\\Test_Image\\NM", vecFiles );
	// _ScanFiles( "D:\\LGCBTray\\Test_Image\\sz_2", vecFiles );
	
	//_ScanFiles("D:\\LGCBTray\\Test_Image\\NM\\1_1", vecFiles);
	// _ScanFiles("D:\\LGCBTray\\Test_Image\\NM\\1_2", vecFiles);
	// _ScanFiles("D:\\LGCBTray\\Test_Image\\NM\\2_1", vecFiles);
	_ScanFiles( "D:\\LGCBTray\\Test_Image\\NM\\2_2", vecFiles );
	

	// _ScanFiles("D:\\LGCBTray\\Test_Image\\NM\\sz1\\水渍原图", vecFiles);
	// _ScanFiles("D:\\LGCBTray\\Test_Image\\NM\\sz1\\1", vecFiles);
	// _ScanFiles("D:\\22", vecFiles);

	// _ScanFiles("E:\\Work\\KD\\4-LG\\03_CB_Tray_Insp\\03_Image\\水渍灰", vecFiles);
	// _ScanFiles( "D:\\LGCBTray\\Test_Image\\NM\\sz", vecFiles );
	
	//std::vector<_TKDWater_Defect>  vecDefect;
	
	char buf[128];
	for ( int idx = 0; idx < vecFiles.size(); ++idx )
	{
		cv::Mat img = cv::imread( vecFiles[idx], cv::IMREAD_GRAYSCALE );
		water.FindDefect( vecDefect, img, 3 );

		cv::Mat imgColor;
		cv::cvtColor( img, imgColor, cv::COLOR_GRAY2BGR );

		for ( int idx = 0; idx < vecDefect.size(); ++idx )
		{
			cv::Rect boxRT = vecDefect[idx].rtRect;
			boxRT.x -= 5;
			boxRT.y -= 5;
			boxRT.width += 10;
			boxRT.height += 10;
			if (boxRT.x < 0) { boxRT.x = 0; }
			if (boxRT.y < 0) { boxRT.y = 0; }
			if (boxRT.x + boxRT.width > imgColor.cols) { boxRT.width = imgColor.cols - boxRT.x; }
			if (boxRT.y + boxRT.height > imgColor.rows) { boxRT.height = imgColor.rows - boxRT.y; }

			cv::rectangle( imgColor, boxRT, cv::Scalar(0, 0, 255), 2 );

			sprintf_s( buf, "%d", vecDefect[idx].iArea );
			cv::putText(imgColor, buf, boxRT.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,0), 2 );
		}
		
		if ( vecDefect.size() > 0 )
		{
			sprintf_s(buf, "d:\\1\\%d.bmp", idx);
			cv::imwrite(buf, imgColor);
		}
		else
		{
			sprintf_s(buf, "d:\\2\\%d.bmp", idx);
			cv::imwrite(buf, imgColor);
		}
	}

}

#endif



#if 0

#include "..\KDLib_Barcode\CBarcode.h"

#include <omp.h>
void _Test_Barcode()
{
	std::vector<std::string>  vecFiles;
	_ScanFiles("E:\\Work\\KD\\4-LG\\03_CB_Tray_Insp\\03_Image\\03NumRec\\", vecFiles);

	CKDLib_Barcode128 code[4];

	char m_strResult[48];

#pragma omp parallel for num_threads(4)
	for (int idx = 0; idx < vecFiles.size(); ++idx)
	{
		char buf[128];
		int iThreadIdx = omp_get_thread_num();
		sprintf_s(buf, "%d --%d", idx, iThreadIdx);
		OutputDebugStringA(buf);

		cv::Mat img = cv::imread(vecFiles[idx], cv::IMREAD_GRAYSCALE);

		std::string strCode = "Error";
		//CBarcode128 code;
		//CBarcode128 code;
		bool ret = code[iThreadIdx].ScanCode(strCode, img);
		// omp_set_num_threads(0);

		cv::Mat smallImg;
		cv::resize(img, smallImg, cv::Size(img.cols >> 1, img.rows >> 1));
		if (true == ret)
		{
			cv::putText(smallImg, strCode, cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar::all(255), 2);

			memset(m_strResult, 0, sizeof(m_strResult));
			memcpy_s(m_strResult, sizeof(m_strResult), strCode.c_str(), strCode.size() - 1);
		}
		else
		{
			strCode = "Error";
			cv::putText(smallImg, strCode, cv::Point(100, 100), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar::all(255), 2);
		}

		sprintf_s(buf, "%d : %s - %s\n", idx, strCode.c_str(), vecFiles[idx].c_str());
		OutputDebugStringA(buf);
	}
}


#endif



// 用于应用程序“关于”菜单项的 CAboutDlg 对话框

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 支持

// 实现
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CTestDlgDlg 对话框



CTestDlgDlg::CTestDlgDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx( IDD_TESTDLG_DIALOG, pParent )
	, m_iValue( 0 )
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}


CTestDlgDlg::~CTestDlgDlg()
{
	;
}


void CTestDlgDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CTestDlgDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_WM_TIMER()
	ON_BN_CLICKED(IDC_BUTTON1, &CTestDlgDlg::OnBnClickedButton1)
	ON_BN_CLICKED(IDC_BUTTON2, &CTestDlgDlg::OnBnClickedButton2)
	ON_BN_CLICKED(IDC_BUTTON3, &CTestDlgDlg::OnBnClickedButton3)
	ON_BN_CLICKED(IDC_BUTTON4, &CTestDlgDlg::OnBnClickedButton4)
END_MESSAGE_MAP()





// CTestDlgDlg 消息处理程序

CTestDlgDlg* g_pMainDlg = NULL;


BOOL CTestDlgDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	g_pMainDlg = this;

	//_Test_Barcode();

	//return TRUE;

	// 将“关于...”菜单项添加到系统菜单中。
	
	// IDM_ABOUTBOX 必须在系统命令范围内。
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 设置此对话框的图标。  当应用程序主窗口不是对话框时，框架将自动
	//  执行此操作
	SetIcon(m_hIcon, TRUE);			// 设置大图标
	SetIcon(m_hIcon, FALSE);		// 设置小图标

	// TODO: 在此添加额外的初始化代码
	//_TestFront();

	//_Test_Water();

	// _Test_Back();


	//KDLog_Start( _T("Log"), _T("Temp"), 10);
	//SetTimer(0, 100, NULL);

	return TRUE;  // 除非将焦点设置到控件，否则返回 TRUE
}


void CTestDlgDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 如果向对话框添加最小化按钮，则需要下面的代码
//  来绘制该图标。  对于使用文档/视图模型的 MFC 应用程序，
//  这将由框架自动完成。

void CTestDlgDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 用于绘制的设备上下文

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 使图标在工作区矩形中居中
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 绘制图标
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

//当用户拖动最小化窗口时系统调用此函数取得光标
//显示。
HCURSOR CTestDlgDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}



void CTestDlgDlg::OnTimer(UINT_PTR nIDEvent)
{
	// TODO: 在此添加消息处理程序代码和/或调用默认值

	UpdateData(FALSE);

	CDialogEx::OnTimer(nIDEvent);
}



// 内面污染  漏检率 数据确认，
// 水渍模板检测的 不良特征分类
// 水渍模板 测试软件 







#include <io.h>
#include <direct.h>

// path
void _ScanFiles_new(std::string path, std::vector<std::string>& vecfiles)
{
	//文件句柄  
	__int64   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;

	std::string affix = "\\*.*";
	if (path.size() <= 0)
	{
		affix = "*.*";
	}

	if ((hFile = _findfirst(p.assign(path).append(affix).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{   // 子文件夹
				if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))
				{  // 非. 和 .. ,即为子目录
					if (path.size() <= 0)
					{
						_ScanFiles(fileinfo.name, vecfiles);
					}
					else
					{
						_ScanFiles(path + "\\" + fileinfo.name, vecfiles);
					}
				}
			}
			else
			{   // 文件
				std::string name = fileinfo.name;
				size_t pos0 = name.find(".jpg");
				size_t pos1 = name.find(".png");
				size_t pos2 = name.find(".bmp");
				if ((pos0 >= 0) || (pos1 >= 0) || (pos2 >= 0))
					// if ((pos0 >= 0) || (pos2 >= 0))
				{
					if (path.size() <= 0)
					{
						vecfiles.push_back(fileinfo.name);
					}
					else
					{
						vecfiles.push_back(path + "\\" + fileinfo.name);
					}
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}








void _ScanDirection(std::string path, std::vector<std::string>& vecfiles)
{
	//文件句柄  
	__int64   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	std::string p;

	if ((hFile = _findfirst(p.assign(path).append("\\*.*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{   // 子文件夹
				if ((strcmp(fileinfo.name, ".") != 0) && (strcmp(fileinfo.name, "..") != 0))
				{  // 非. 和 .. ,即为子目录
					_ScanDirection(path + "\\" + fileinfo.name, vecfiles);
					vecfiles.push_back(path + "\\" + fileinfo.name);
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}


void _Delete_Path(std::string path)
{
	std::vector<std::string> vecFilePath;

	//------------------< 1. 删除文件 >-----------
	// 获取路径及子文件夹内的  所有文件

	vecFilePath.clear();
	_ScanFiles_new(path, vecFilePath);

	size_t iLen = vecFilePath.size();

	for (int idx = 0; idx < iLen; ++idx)
	{
		DeleteFileA(vecFilePath[idx].c_str());
	}

	//------------------< 2. 删除文件夹 >-----------

	vecFilePath.clear();
	_ScanDirection(path, vecFilePath);
	iLen = vecFilePath.size();
	for (int idx = 0; idx < iLen; ++idx)
	{
		RemoveDirectoryA(vecFilePath[idx].c_str());
	}
}



void _CreateFolder(std::string filePath)
{
	size_t iLen = filePath.size();
	if (iLen <= 2)
	{
		return;
	}

	std::string path = filePath;
	size_t dotPos = path.rfind('.');
	if ((dotPos < 0) && (path[iLen - 1] != '\\'))
	{
		path += '\\';
	}

	__int64 lastPos = path.rfind('\\');
	if (lastPos < 0)
	{
		return;
	}

	path = path.substr(0, lastPos + 1);
	if (-1 != _access(path.c_str(), 0))
	{
		return;
	}

	__int64 m = 0, n = 0;
	std::string str1, str2;
	str1 = path;
	str2 = "";

	while (1)
	{
		m = str1.find('\\');
		if (m < 0)
		{
			break;
		}

		str2 += str1.substr(0, m) + '\\';

		// 判断该目录是否存在
		n = _access(str2.c_str(), 0);
		if (n == -1)
		{   //创建目录文件
			_mkdir(str2.c_str());
		}

		str1 = str1.substr(m + 1, str1.size());
	}
}






#include "CAlgo_KDWater2.h"

// -d yolov5_7.0_LG.engine ./images classes.txt
void CheckWater_Common( std::string filePath, std::string strResult, int iDir )
{
	std::string strResultAll = "ResultAll\\";
	_Delete_Path(strResult);
	_CreateFolder(strResult);

//	_Delete_Path(strResultAll);
	_CreateFolder(strResultAll);

	_Delete_Path("OK\\" + filePath + "\\");
	_Delete_Path("NG\\" + filePath + "\\");
	_CreateFolder("OK\\" + filePath + "\\");
	_CreateFolder("NG\\" + filePath + "\\");

	std::vector<std::string> vecPath;
	_ScanFiles_NoPath(filePath, vecPath);

	CAlgo_KDWater2 algo;
	algo.LoadTemplate();

	for ( int idx = 0; idx < vecPath.size(); ++idx )
	{
		cv::Mat img = cv::imread( filePath + "\\" + vecPath[idx], cv::IMREAD_GRAYSCALE );
		cv::Mat imgColor = cv::imread(filePath + "\\" + vecPath[idx], cv::IMREAD_COLOR);
		cv::Mat img_org;
		img.copyTo(img_org);
		std::vector<_TKDWater_Defect2> vecDefect;
		bool result = algo.FindDefect(vecDefect, img, iDir, imgColor);

		if (result == false)
		{
			std::string filename1 = "NG\\" + filePath + "\\" + vecPath[idx];
			cv::imwrite(filename1, imgColor);

			filename1.replace(filename1.size() - 3, 3, "txt");
			std::ofstream MyFile(filename1);
			char buf[128];
			for (int idx = 0; idx < vecDefect.size(); ++idx)
			{
				cv::Rect boxRT = vecDefect[idx].rtRect;
				boxRT.x -= 0;
				boxRT.y -= 0;
				boxRT.width += 0;
				boxRT.height += 0;
				if (boxRT.x < 0) { boxRT.x = 0; }
				if (boxRT.y < 0) { boxRT.y = 0; }
				if (boxRT.x + boxRT.width > imgColor.cols) { boxRT.width = imgColor.cols - boxRT.x; }
				if (boxRT.y + boxRT.height > imgColor.rows) { boxRT.height = imgColor.rows - boxRT.y; }

				cv::rectangle(imgColor, boxRT, cv::Scalar(0, 0, 255), 2);

				//sprintf_s(buf, "%.2f", vecDefect[idx].iArea*0.141*0.141);
				sprintf_s(buf, "%d", vecDefect[idx].iArea);
				cv::putText(imgColor, buf, boxRT.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

				sprintf_s(buf, "%d", vecDefect[idx].iMeanVal);
				cv::putText(imgColor, buf, boxRT.tl() + cv::Point(0,boxRT.height+16) , cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
				std::vector<std::vector<cv::Point>> contour;
				contour = vecDefect[idx].contour;

				for (size_t i = 0; i < contour.size(); i++) {
					MyFile << "0";
					for (size_t j = 0; j < contour[i].size(); j++) {
						// Write contour points to the file
						MyFile << " " << (contour[i][j].x + vecDefect[idx].p0.x)/2448.0 << " " << (contour[i][j].y + vecDefect[idx].p0.y)/2048.0;
					}
					// Separate contours by a blank line
					MyFile << "\n";
				}

			}
			MyFile.close();
			std::string filename = strResult + "\\" + vecPath[idx];
			filename.replace(filename.size() - 3, 3, "jpg");
			cv::imwrite(filename, imgColor);
			filename = strResultAll + "\\" + vecPath[idx];
			filename.replace(filename.size() - 3, 3, "jpg");
			cv::imwrite(filename, imgColor);
		}
		else {
			std::string filename = "OK\\" + filePath + "\\" + vecPath[idx];
			filename.replace(filename.size() - 3, 3, "jpg");
			cv::imwrite(filename, imgColor);
		}
	}
}



//std::string g_strPath = "D:\\LGCBTray\\Test_Image\\NM\\";
//std::string g_strPath = "D:\\LGCBTray\\Test_Image\\NM\\0606\\";
//std::string g_strPath = "D:\\1\\";
std::string g_strPath = "";

void CheckWater_1_1( void )
{
	std::string filePath = g_strPath + "1_1";
	std::string strResult = g_strPath + "Result_1_1\\";

	CheckWater_Common(filePath, strResult, 0);
	//BMP2JPG(filePath, strResult);

	g_pMainDlg->GetDlgItem(IDC_BUTTON1)->EnableWindow(TRUE);
}


// 1 - 1
void CTestDlgDlg::OnBnClickedButton1()
{
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_BUTTON1)->EnableWindow(FALSE);
	std::thread th(CheckWater_1_1);
	th.detach();
}


void CheckWater_1_2(void)
{
	std::string filePath = g_strPath + "1_2";
	std::string strResult = g_strPath + "Result_1_2\\";

	CheckWater_Common(filePath, strResult, 1);
	//BMP2JPG(filePath, strResult);

	g_pMainDlg->GetDlgItem(IDC_BUTTON2)->EnableWindow(TRUE);
}


// 1 - 2
void CTestDlgDlg::OnBnClickedButton2()
{
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_BUTTON2)->EnableWindow(FALSE);
	std::thread th(CheckWater_1_2);
	th.detach();
}


void CheckWater_2_1(void)
{
	std::string filePath = g_strPath + "2_1";
	std::string strResult = g_strPath + "Result_2_1\\";

	CheckWater_Common(filePath, strResult, 2);
	//BMP2JPG(filePath, strResult);

	g_pMainDlg->GetDlgItem(IDC_BUTTON3)->EnableWindow(TRUE);
}

// 2 - 1
void CTestDlgDlg::OnBnClickedButton3()
{
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_BUTTON3)->EnableWindow(FALSE);
	std::thread th(CheckWater_2_1);
	th.detach();
}


void CheckWater_2_2( void )
{
	std::string filePath = g_strPath + "2_2";
	std::string strResult = g_strPath + "Result_2_2\\";

	CheckWater_Common(filePath, strResult, 3);
	//BMP2JPG(filePath, strResult);

	g_pMainDlg->GetDlgItem(IDC_BUTTON4)->EnableWindow(TRUE);
}

// 2 - 2
void CTestDlgDlg::OnBnClickedButton4()
{
	// TODO: 在此添加控件通知处理程序代码
	GetDlgItem(IDC_BUTTON4)->EnableWindow(FALSE);
	std::thread th(CheckWater_2_2);
	th.detach();
}

