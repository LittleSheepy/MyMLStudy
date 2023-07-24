
// TestDlgDlg.h: 头文件
//

#pragma once

//#include "Ctrl_ImageView.h"

//#include "CAlgo.h"
#include "resource.h"
#include <afxwin.h>
#include <afxmt.h>


// CTestDlgDlg 对话框
class CTestDlgDlg : public CDialogEx
{
// 构造
public:
	CTestDlgDlg(CWnd* pParent = nullptr);	// 标准构造函数

	~CTestDlgDlg();

private:

// 对话框数据
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_TESTDLG_DIALOG };
#endif

protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 支持


// 实现
protected:
	HICON m_hIcon;

	// 生成的消息映射函数
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg void OnTimer(UINT_PTR nIDEvent);
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

public:

	int m_iValue;
	afx_msg void OnBnClickedButton1();
	afx_msg void OnBnClickedButton2();
	afx_msg void OnBnClickedButton3();
	afx_msg void OnBnClickedButton4();
};
