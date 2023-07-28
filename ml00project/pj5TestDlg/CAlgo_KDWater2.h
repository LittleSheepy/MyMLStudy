#pragma once

#include <opencv.hpp>

#include "yolov5/include/IYoloSeg.h"

typedef struct
{
	cv::Rect  rtRect;
	cv::Point p0;
	int       iPos;
	int       iArea;
	int       iMeanVal;  // 平均亮度

	bool bNIC;
	cv::Point ptCenter;
	std::vector<std::vector<cv::Point>> contour;
}_TKDWater_Defect2;

int calculate_mean(const cv::Mat& gray_image, const cv::Point& center, int diameter = 30);

typedef struct  
{
	int   iThresd_BlobArea;
	int   iThresd_Diff;
	int   iThresd_PixelValue_Water;  // 水渍像素
	int   iThresd_MaxMeanValue;   // 水渍区域亮度最大值
}_TKDWater_Param;


class _CKDWater_MatchBox
{
public:

	_CKDWater_MatchBox()
	{
		m_iMeanVal = 80;
		m_iRow = 0;
		m_iCol = 0;
		m_vptNIC.push_back(cv::Point(124, 124));

		m_vptNIC.push_back(cv::Point(65, 100));
		m_vptNIC.push_back(cv::Point(65, 148));
		m_vptNIC.push_back(cv::Point(100, 183));
		m_vptNIC.push_back(cv::Point(148, 183));

		m_vptNIC.push_back(cv::Point(183, 148));
		m_vptNIC.push_back(cv::Point(183, 100));
		m_vptNIC.push_back(cv::Point(148, 65));
		m_vptNIC.push_back(cv::Point(100, 65));
	}

	bool IsPtInRect( cv::Point& pt )
	{
		bool ret = false;
		int iHalfW = imgMatch.cols >> 1;
		int iHalfH = imgMatch.rows >> 1;
		
		if ( (pt.x >= ptCent.x - iHalfW)
			&& (pt.y >= ptCent.y - iHalfH)
			&& (pt.x < ptCent.x + iHalfW)
			&& (pt.y < ptCent.y + iHalfH)
			)
		{
			ret = true;
		}
		
		return ret;
	}

	void CalcMeanVal01( )
	{
		////////////////////////////////////////

		cv::blur(imgMatch, imgMatch, cv::Size(5, 5));

 		//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
		//cv::morphologyEx(imgMatch, imgMatch, cv::MORPH_CLOSE, kernel);
		
		//cv::morphologyEx(imgMatch, imgMatch, cv::MORPH_OPEN, kernel);
 		//cv::morphologyEx(imgMatch, imgMatch, cv::MORPH_CLOSE, kernel);

// 		cv::Mat imgBin;
// 		cv::threshold(imgMatch, imgBin, 0, 255, cv::THRESH_OTSU);
// 		cv::Scalar val = cv::mean(imgMatch, imgBin);

		cv::Mat imgRoi = imgMatch(cv::Rect(imgMatch.cols >> 2, imgMatch.rows >> 2, imgMatch.cols >> 1, imgMatch.rows >> 1));
		//cv::Mat imgBin;
		//cv::threshold(imgRoi, imgBin, 0, 255, cv::THRESH_OTSU);
		//cv::Scalar val = cv::mean(imgRoi, imgBin);

		cv::Scalar val = cv::mean(imgRoi);
		if (val[0] < 60.0f)
		{
			val[0] = 60;
		}

		m_iMeanVal = (int)val[0];

		double rate = 160.0f / val[0];

		imgMatch *= rate;
	}


	void CalcMeanVal( int iThresd_Water )
	{
		cv::blur(imgMatch, imgMatch, cv::Size(5, 5));
		cv::threshold(imgMatch, imgRegionWater, iThresd_Water, 255, cv::THRESH_BINARY_INV );
		
		cv::Rect roiRect(28, 28, imgMatch.cols - 56, imgMatch.rows - 56);
		cv::Mat roiImg = imgMatch(roiRect);
		
		std::vector<double> vecPixel;
		for ( int row = 0; row < roiImg.rows; row += 2 )
		{
			uchar* pLine = roiImg.ptr<uchar>(row);
			for ( int col = 0; col < roiImg.cols; col += 2  )
			{
				vecPixel.push_back(pLine[col]);
			}
		}
		
		double dStdDev = 0.0;
		double dStdMean = 0.0;
		stddev(&dStdDev, &dStdMean, vecPixel);

		double dScale = 50.0f / dStdDev;
		double dOffs = 100 - dStdMean * dScale;

		imgMatch *= dScale;
		imgMatch += dOffs;

		///////////////////////////////////////////////
		for (int i = 0; i < 9; ++i) {
			cv::Point nic_center_pt = m_vptNIC[i];

			int nic_offset_x = static_cast<int>((3 - m_iCol) * 1.7);
			int nic_offset_y = static_cast<int>((3 - m_iRow) * 1.7);
			nic_center_pt.x += nic_offset_x;
			nic_center_pt.y += nic_offset_y;

			int mean = calculate_mean(imgMatch, nic_center_pt);
			m_nic_mean.push_back(mean);
		}
	}

	double mean( std::vector <double>& data) 
	{
		double ans = 0.0;
		for (int i = 0; i < data.size(); i++) 
		{
			ans += data[i];
		}
		ans = ans / data.size();
		return ans;
	}

	void stddev(double *pdDev, double *pdMean, std::vector <double>& data )
	{
		double a1 = mean(data);
		double a2 = 0.0;
		for (int i = 0; i < data.size(); i++) 
		{
			a2 += (a1 - data[i]) * (a1 - data[i]);
		}
		a2 = sqrt(a2 / (data.size() - 1) );
		
		*pdMean = a1;
		*pdDev = a2;
	}

public:
	
	int        m_iRow;
	int        m_iCol;
	int        m_iMeanVal;

	cv::Point  ptCent;
	cv::Mat    imgMatch;
	cv::Mat    imgColor;

	cv::Mat   imgRegionWater;

	cv::Mat    imgBoxFull;
	std::vector<cv::Point> m_vptNIC;
	std::vector<int> m_nic_mean;
};


class CAlgo_KDWater2
{
public:

	CAlgo_KDWater2( )
	{
		m_tParam.iThresd_BlobArea = 60;  // 水渍最小面积值
		m_tParam.iThresd_Diff = 30;  // 水渍与其他区域的对比最小值

		m_tParam.iThresd_MaxMeanValue = 70;  // 水渍区域平均亮度最大值
		m_tParam.iThresd_PixelValue_Water = 90; // 水渍像素值
		yoloseg = createYoloSegInstance();
		yoloseg->SetEngineName(engine_name);
		yoloseg->Init();

	}
	
	~CAlgo_KDWater2( )
	{
		;
	}

	void SetParam( int iThresd_Diff, int iThresd_BlobArea )
	{
		m_tParam.iThresd_Diff = iThresd_Diff;
		m_tParam.iThresd_BlobArea = iThresd_BlobArea;
	}
	
	bool LoadTemplate( );
	
	// 0 - LT, 1 - LB,  2 - RT,  3 - RB
	bool FindDefect( std::vector<_TKDWater_Defect2>& vecDefect, cv::Mat srcImgGray, int iDir, cv::Mat srcImgColor);


private:
	
	bool _GetBestMatchPos( cv::Point& cent, cv::Mat& srcImgGray );

	void _SearchDefect( std::vector<_TKDWater_Defect2>& vecDefect, _CKDWater_MatchBox &box, cv::Mat &srcImgGray, int iDir );
	void _SearchBox_Neib( std::vector<_CKDWater_MatchBox>& vecBox, _CKDWater_MatchBox& curBox, cv::Mat& srcImgGray );

	bool _GetNeibBox( _CKDWater_MatchBox &curBox, cv::Point offsPt, std::vector<_CKDWater_MatchBox>& vecBox, cv::Mat &srcImgGray );


private:

	void _CheckSpecTab( cv::Mat &imgBin, _CKDWater_MatchBox &curBox, int iDir );

	void _GetDefect_Neib( std::vector<_TKDWater_Defect2>& vecDefect, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat &srcImgGray );
	void _GetDefect_Neib2( std::vector<_TKDWater_Defect2>& vecDefect, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat &srcImgGray );
	
	void _GetDefectImgBin_Normal(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow);
	void _GetDefectImgBin_Col0(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat &srcImgGray);
	void _GetDefectImgBin_Row0(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat& srcImgGray);

	void _GetDefectImgBin_Side(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow );
	
	void _GetDefectImgBin_Neib34(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iNeibIndex[4]);
	void _GetDefectImgBin_Neib2(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iNeibIndex[2]);
	void _GetDefectImgBin_Neib_Least2(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, std::vector<int> &vecIndex );
	
	bool _CheckDefect_ByValue( int &iMeanVOut, cv::Mat& imgCur, std::vector<std::vector<cv::Point>>& vecCont, int iContIdx );
	bool _CheckDefect_ByRound( cv::Mat& imgCur, cv::Rect& rtDefect, int iThresd_NeibDiff );

private:

	bool _GetMaxMatch( _CKDWater_MatchBox &outBox, cv::Mat& roiImg, cv::Mat& imgTempl );


	// 获取二条直线交点
	cv::Point2d _get2lineIPoint(cv::Vec4f lineParam1, cv::Vec4f lineParam2)
	{
		//Vec4f :参数的前半部分给出的是直线的方向，而后半部分给出的是直线上的一点

#define EPS 1e-8

		cv::Point2d result(-1, -1);

		double cos_theta = lineParam1[0];
		double sin_theta = lineParam1[1];
		if (cos_theta < EPS)
		{
			cos_theta = EPS;
		}

		double x = lineParam1[2];
		double y = lineParam1[3];
		double k = sin_theta / cos_theta;
		double b = y - k * x;

		cos_theta = lineParam2[0];
		sin_theta = lineParam2[1];
		x = lineParam2[2];
		y = lineParam2[3];
		double k1 = sin_theta / cos_theta;
		double b1 = y - k1 * x;

		result.x = (b1 - b) / (k - k1);
		result.y = k * result.x + b;

		return result;
	}


private:

	// 统一排序
	void _UniteSortRowCol( std::vector<_CKDWater_MatchBox>& vecBox, int iDir );

	// 补充完整周边的格子
	void _CompleteRoundBox(std::vector<_CKDWater_MatchBox>& vecBox, cv::Mat& srcImgGray, int iDir );

private:

	cv::Mat    m_imgTempl;
	cv::Mat    m_imgTempl_tabMask_Col;
	cv::Mat    m_imgTempl_tabMask_Row;
	
	cv::Mat    m_imgRegion_Inner;    // 内部区域
	cv::Mat    m_imgRegion_Outter;   // 棱边

	cv::Point  m_ptStdOffs;

private:

	_TKDWater_Param   m_tParam;
	std::vector<cv::Point> m_vptNIC;
	cv::Mat img_color;

	IYoloSeg* yoloseg = NULL;
	std::string wts_name = "";
	//std::string engine_name = "yolov5_7.0_LG_NMSZ.engine";
	std::string engine_name = "seg_best_nmjqjps.engine";
	bool is_p6 = false;
	float gd = 0.33f, gw = 0.50f;
	std::string img_dir = "./images";
};

