#include "pch.h"
#include "CAlgo_KDWater2.h"
#include <fstream>
#include <ostream>
#include "yolov5/include/IYoloSeg.h"

#define DEF_THRESD_AREA     40   // 
#define DEF_THRESD_DIFF     30    //  -- 水渍与其他区域的对比最小值

int calculate_mean(const cv::Mat& gray_image, const cv::Point& center, int diameter) {
	int radius = diameter / 2;
	cv::Mat mask = cv::Mat::zeros(gray_image.size(), CV_8U);
	cv::circle(mask, center, radius, 255, -1);
	cv::Mat masked_image;
	cv::bitwise_and(gray_image, gray_image, masked_image, mask);
	cv::Scalar mean_value = cv::mean(masked_image, mask);
	return (int)mean_value[0];
}
bool CAlgo_KDWater2::LoadTemplate( )
{
	m_imgTempl = cv::imread( "template\\nm_block.bmp", cv::IMREAD_GRAYSCALE );
	if ( !m_imgTempl.data )
	{
		return false;
	}

	m_ptStdOffs = cv::Point(213, 213);
	m_imgTempl_tabMask_Col = cv::imread( "template\\nm_tabmask_col_0607.bmp", cv::IMREAD_GRAYSCALE );
	m_imgTempl_tabMask_Row = cv::imread("template\\nm_tabmask_row_0607.bmp", cv::IMREAD_GRAYSCALE);

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::erode(m_imgTempl_tabMask_Col, m_imgTempl_tabMask_Col, kernel);
	cv::erode(m_imgTempl_tabMask_Row, m_imgTempl_tabMask_Row, kernel);


	if ( (!(m_imgTempl_tabMask_Row.data)) 
		|| (m_imgTempl_tabMask_Row.size() != m_imgTempl.size()) )
	{
		m_imgTempl_tabMask_Row = cv::Mat::zeros(m_imgTempl.size(), CV_8UC1);
		m_imgTempl_tabMask_Row.setTo(255);
	}

	if ((!(m_imgTempl_tabMask_Col.data))
		|| (m_imgTempl_tabMask_Col.size() != m_imgTempl.size()))
	{
		m_imgTempl_tabMask_Col = cv::Mat::zeros(m_imgTempl.size(), CV_8UC1);
		m_imgTempl_tabMask_Col.setTo(255);
	}

	///////////////////////////////////////
	
	//int iWid_In = 30;
	int iWid_In = 36;

	m_imgRegion_Inner = cv::Mat::zeros(m_imgTempl.size(), CV_8UC1);
	//cv::rectangle(m_imgRegion_Inner, cv::Rect( iWid_In, iWid_In, m_imgTempl.cols - iWid_In * 2, m_imgTempl.rows - iWid_In * 2), cv::Scalar::all(255), -1);
	cv::circle(m_imgRegion_Inner, cv::Point(m_imgTempl.cols >> 1, m_imgTempl.rows >> 1), (m_imgTempl.cols >> 1) - 30, cv::Scalar::all(255), -1 );

	m_imgRegion_Outter = 255 - m_imgRegion_Inner;



	//{
	//	Json::Value root;
	//	Json::Reader reader;

	//	std::string filepath = "device.cfg";

	//	std::ifstream cfgfile(filepath);
	//	std::istreambuf_iterator<char> beg(cfgfile), end;
	//	std::string outStr(beg, end);
	//	cfgfile.close();
	//	if (outStr.size() > 0 )
	//	{
	//		if (reader.parse(outStr, root))
	//		{
	//			m_tParam.iThresd_Diff = root["Thresd_Diff"].asInt();
	//			m_tParam.iThresd_BlobArea = root["Thresd_Area"].asInt();
	//			m_tParam.iThresd_MaxMeanValue = root["Thresd_MaxMeanValue"].asInt();
	//			m_tParam.iThresd_PixelValue_Water = root["Thresd_PixelValue_Water"].asInt();
	//		}
	//	}
	//}

	///////////////////////////////////////////////////////////////

	//{
	//	Json::Value root;
	//	root.clear();

	//	root["Thresd_Diff"] = m_tParam.iThresd_Diff;
	//	root["Thresd_Area"] = m_tParam.iThresd_BlobArea;
	//	root["Thresd_MaxMeanValue"] = m_tParam.iThresd_MaxMeanValue;
	//	root["Thresd_PixelValue_Water"] = m_tParam.iThresd_PixelValue_Water;

	//	std::string filepath = "device.cfg";
	//	std::fstream cfgfile(filepath, std::ios::out);
	//	Json::StyledStreamWriter writer;
	//	writer.write(cfgfile, root);
	//	cfgfile.close();
	//}


	// NIC
	m_vptNIC.push_back(cv::Point(124, 124));

	m_vptNIC.push_back(cv::Point(65, 100));
	m_vptNIC.push_back(cv::Point(65, 148));
	m_vptNIC.push_back(cv::Point(100, 183));
	m_vptNIC.push_back(cv::Point(148, 183));

	m_vptNIC.push_back(cv::Point(183, 148));
	m_vptNIC.push_back(cv::Point(183, 100));
	m_vptNIC.push_back(cv::Point(148, 65));
	m_vptNIC.push_back(cv::Point(100, 65));


// 	cv::Mat img[4];
// 	cv::flip(m_imgTempl_tabMask, img[0], 0);
// 	cv::flip(m_imgTempl_tabMask, img[1], 1);
// 	cv::flip(m_imgTempl_tabMask, img[2], -1);
// 	cv::flip(img[2], img[3], 0);
	return true;
}


// 0 - LT, 1 - LB,  2 - RT,  3 - RB
bool CAlgo_KDWater2::FindDefect( std::vector<_TKDWater_Defect2>& vecDefect, cv::Mat srcImgGray, int iDir, cv::Mat srcImgColor)
{
	img_color = srcImgColor;
	vecDefect.clear( );
	if ( !(m_imgTempl.data) || !(m_imgTempl_tabMask_Col.data) || !(m_imgTempl_tabMask_Row.data) )
	{
		return false;
	}

	cv::Point ptCent;
	cv::Rect rtCent( m_imgTempl.cols, m_imgTempl.rows,  srcImgGray.cols - m_imgTempl.cols*2, srcImgGray.rows - m_imgTempl.rows * 2 );

	cv::GaussianBlur( srcImgGray, srcImgGray, cv::Size(3, 3), 5.0f );

	if ( _GetBestMatchPos(ptCent, srcImgGray(rtCent)) )
	{
		_CKDWater_MatchBox  box;
		box.ptCent = ptCent + rtCent.tl();

		cv::Rect curRect(box.ptCent.x - (m_imgTempl.cols >> 1), box.ptCent.y - (m_imgTempl.rows >> 1), m_imgTempl.cols, m_imgTempl.rows);
		srcImgGray(curRect).copyTo( box.imgMatch );
		cv::Rect curRect1(box.ptCent.x - (m_imgTempl.cols >> 1), box.ptCent.y - (m_imgTempl.rows >> 1), m_imgTempl.cols, m_imgTempl.rows);
		srcImgColor(curRect1).copyTo( box.imgColor);

		box.m_iRow = 0;
		box.m_iCol = 0;
		_SearchDefect( vecDefect, box, srcImgGray, iDir );
	}
	if (vecDefect.size() == 0)
	{
		return true;
	}

	return vecDefect.size() > 0 ? false : true;
}



// 0 - LT, 1 - LB,  2 - RT,  3 - RB
void CAlgo_KDWater2::_UniteSortRowCol( std::vector<_CKDWater_MatchBox>& vecBox, int iDir )
{
	int iTotal = vecBox.size();
	if ( iTotal <= 0 )
	{
		return;
	}

	int iMinX = vecBox[0].m_iCol;
	int iMaxX = vecBox[0].m_iCol;
	int iMinY = vecBox[0].m_iRow;
	int iMaxY = vecBox[0].m_iRow;

	for ( int idx = 1; idx < iTotal; ++idx )
	{
		int iCurX = vecBox[idx].m_iCol;
		int iCurY = vecBox[idx].m_iRow;
		
		if (iCurX < iMinX) { iMinX = iCurX; }
		if (iCurY < iMinY) { iMinY = iCurY; }
		if (iCurX > iMaxX) { iMaxX = iCurX; }
		if (iCurY > iMaxY) { iMaxY = iCurY; }
	}

	switch ( iDir )
	{
	case 0:  // LT
		for (int idx = 0; idx < iTotal; ++idx)
		{
			vecBox[idx].m_iCol = vecBox[idx].m_iCol - iMinX;
			vecBox[idx].m_iRow = vecBox[idx].m_iRow - iMinY;
		}
		break;
	case 1:  // LB
		for (int idx = 0; idx < iTotal; ++idx)
		{
			vecBox[idx].m_iCol = vecBox[idx].m_iCol - iMinX;
			vecBox[idx].m_iRow = iMaxY - vecBox[idx].m_iRow;
		}
		break;
	case 2:  // RT
		for (int idx = 0; idx < iTotal; ++idx)
		{
			vecBox[idx].m_iCol = iMaxX - vecBox[idx].m_iCol;
			vecBox[idx].m_iRow = vecBox[idx].m_iRow - iMinY;
		}
		break;

	default: // RB
		for (int idx = 0; idx < iTotal; ++idx)
		{
			vecBox[idx].m_iCol = iMaxX - vecBox[idx].m_iCol;
			vecBox[idx].m_iRow = iMaxY - vecBox[idx].m_iRow;
		}
		break;
	}
}




inline int _GetBoxByRowCol( std::vector<_CKDWater_MatchBox>& vecBox, int ix, int iy )
{
	int ret = -1;
	int iTotal = vecBox.size();
	for (int idx = 0; idx < iTotal; ++idx)
	{
		if ((vecBox[idx].m_iCol == ix) && (vecBox[idx].m_iRow == iy))
		{
			ret = idx;
			break;
		}
	}
	return ret;
}


inline void _CopyImage2Box( _CKDWater_MatchBox& box, cv::Mat& srcImgGray )
{
	int iHalfW = box.imgMatch.cols >> 1;
	int iHalfH = box.imgMatch.rows >> 1;

	int iStX_Src = box.ptCent.x - iHalfW;
	int iStY_Src = box.ptCent.y - iHalfH;
	int iEndX_Src = iStX_Src + box.imgMatch.cols;
	int iEndY_Src = iStY_Src + box.imgMatch.rows;
	int iStX_Dst = 0;
	int iStY_Dst = 0;
	if (iStX_Src < 0)
	{
		iStX_Dst = -iStX_Src;
		iStX_Src = 0;
	}
	if (iStY_Src < 0)
	{
		iStY_Dst = -iStY_Src;
		iStY_Src = 0;
	}
	if (iEndX_Src > srcImgGray.cols)
	{
		iEndX_Src = srcImgGray.cols;
	}
	if (iEndY_Src > srcImgGray.rows)
	{
		iEndY_Src = srcImgGray.rows;
	}

	cv::Rect srcRect(iStX_Src, iStY_Src, iEndX_Src - iStX_Src, iEndY_Src - iStY_Src);
	cv::Rect dstRect(iStX_Dst, iStY_Dst, srcRect.width, srcRect.height);
	srcImgGray(srcRect).copyTo(box.imgMatch(dstRect) );
}


inline void _CopyImage2Box_0607( _CKDWater_MatchBox& box, cv::Mat& srcImgGray, cv::Mat &imgRefer, int iWid, int iHei )
{
	if ( imgRefer.data )
	{
		imgRefer.copyTo(box.imgMatch);
	}
	else
	{
		box.imgMatch = cv::Mat::zeros(cv::Size(iWid, iHei), CV_8UC1);
	}

	int iHalfW = box.imgMatch.cols >> 1;
	int iHalfH = box.imgMatch.rows >> 1;

	int iStX_Src = box.ptCent.x - iHalfW;
	int iStY_Src = box.ptCent.y - iHalfH;
	int iEndX_Src = iStX_Src + box.imgMatch.cols;
	int iEndY_Src = iStY_Src + box.imgMatch.rows;
	int iStX_Dst = 0;
	int iStY_Dst = 0;
	if (iStX_Src < 0)
	{
		iStX_Dst = -iStX_Src;
		iStX_Src = 0;
	}
	if (iStY_Src < 0)
	{
		iStY_Dst = -iStY_Src;
		iStY_Src = 0;
	}
	if (iEndX_Src > srcImgGray.cols)
	{
		iEndX_Src = srcImgGray.cols;
	}
	if (iEndY_Src > srcImgGray.rows)
	{
		iEndY_Src = srcImgGray.rows;
	}

	cv::Rect srcRect(iStX_Src, iStY_Src, iEndX_Src - iStX_Src, iEndY_Src - iStY_Src);
	cv::Rect dstRect(iStX_Dst, iStY_Dst, srcRect.width, srcRect.height);
	srcImgGray(srcRect).copyTo(box.imgMatch(dstRect));
}

// 0 - LT,  1 - LB,  2 - RT,  3 - RB
void CAlgo_KDWater2::_CompleteRoundBox( std::vector<_CKDWater_MatchBox>& vecBox, cv::Mat& srcImgGray, int iDir )
{
	int iTotal = vecBox.size();
	if ( iTotal <= 0 )
	{
		return;
	}

	int iMaxRow = 0;
	int iMaxCol = 0;
	for ( int idx = 0; idx < iTotal; ++idx )
	{
		if (vecBox[idx].m_iRow > iMaxRow)
		{
			iMaxRow = vecBox[idx].m_iRow;
		}
		if (vecBox[idx].m_iCol > iMaxCol)
		{
			iMaxCol = vecBox[idx].m_iCol;
		}
	}

	/////////////////////////////////////////////////////
	
	int iRowCnt = iMaxRow + 1;
	int iColCnt = iMaxCol + 1;

	//////////////////////////////////////////////////////

	std::vector<cv::Vec4f> vecLineRow;
	for ( int row = 0; row < iRowCnt; ++row )
	{
		std::vector<cv::Point> vecDot;
		for ( int col = 0; col < iColCnt; ++col )
		{
			int iBoxIdx0 = _GetBoxByRowCol( vecBox, col, row );
			if ( iBoxIdx0 >= 0 )
			{
				vecDot.push_back(vecBox[iBoxIdx0].ptCent);
			}
		}

		if ( vecDot.size() > 3 )
		{
			cv::Vec4f line;
			cv::fitLine(vecDot, line, cv::DistanceTypes::DIST_L2, 0, 0.01, 0.01);
			vecLineRow.push_back(line);
		}
	}

	int iSize = vecLineRow.size();
	cv::Vec4f line_1 = vecLineRow[iSize - 2];
	cv::Vec4f line_0 = vecLineRow[iSize - 1];
	cv::Vec4f line = line_0;
	line[3] += line_0[3] - line_1[3];
	vecLineRow.push_back(line);

	std::vector<cv::Vec4f> vecLineCol;
	for ( int col = 0; col < iColCnt; ++col )
	{
		std::vector<cv::Point> vecDot;
		for ( int row = 0; row < iRowCnt; ++row )
		{
			int iBoxIdx = _GetBoxByRowCol(vecBox, col, row);
			if (iBoxIdx >= 0)
			{
				vecDot.push_back(vecBox[iBoxIdx].ptCent);
			}
		}

		if (vecDot.size() > 3)
		{
			cv::Vec4f line;
			cv::fitLine(vecDot, line, cv::DistanceTypes::DIST_L2, 0, 0.01, 0.01);
			vecLineCol.push_back(line);
		}
	}

	iSize = vecLineCol.size();
	line_1 = vecLineCol[iSize - 2];
	line_0 = vecLineCol[iSize - 1];
	line = line_0;
	line[2] += line_0[2] - line_1[2];
	vecLineCol.push_back(line);


	//////////////////////////////////////////////////////

	vecBox.clear();
	for ( int row = 0; row < vecLineRow.size() - 1; ++row )
	{
		cv::Vec4f  line_row = vecLineRow[row];
		for ( int col = 0; col < vecLineCol.size() - 1; ++col )
		{
			cv::Vec4f  line_col = vecLineCol[col];
			cv::Point cent = _get2lineIPoint(line_row, line_col);

			_CKDWater_MatchBox box;
			box.ptCent = cent;
			box.m_iRow = row;
			box.m_iCol = col;

			cv::Mat img;
			_CopyImage2Box_0607(box, srcImgGray, img, m_imgTempl.cols, m_imgTempl.rows );
			vecBox.push_back(box);
		}

		{
			_CKDWater_MatchBox box;
			box.m_iRow = row;
			box.m_iCol = vecLineCol.size() - 1;
			cv::Vec4f  line_col = vecLineCol[box.m_iCol];
			cv::Point cent = _get2lineIPoint(line_row, line_col);
			box.ptCent = cent;


			int iLastBoxIndex = _GetBoxByRowCol( vecBox, box.m_iCol-1, box.m_iRow );
			if ( iLastBoxIndex < 0 )
			{
				continue;
			}

			_CopyImage2Box_0607(box, srcImgGray, vecBox[iLastBoxIndex].imgMatch, m_imgTempl.cols, m_imgTempl.rows);
			vecBox.push_back(box);
		}
	}

	cv::Vec4f  line_row = vecLineRow[vecLineRow.size() - 1];
	for (int col = 0; col < vecLineCol.size(); ++col)
	{
		_CKDWater_MatchBox box;
		box.m_iRow = vecLineRow.size() - 1;
		box.m_iCol = col;
		cv::Vec4f  line_col = vecLineCol[box.m_iCol];
		cv::Point cent = _get2lineIPoint(line_row, line_col);
		box.ptCent = cent;

		int iLastBoxIndex = _GetBoxByRowCol(vecBox, box.m_iCol, box.m_iRow - 1);
		if (iLastBoxIndex < 0)
		{
			continue;
		}

		_CopyImage2Box_0607(box, srcImgGray, vecBox[iLastBoxIndex].imgMatch, m_imgTempl.cols, m_imgTempl.rows);
		vecBox.push_back(box);
	}


	//////////////////////////////////////////////////////

#if 0
	// 补列
	for ( int idx = 0; idx < iRowCnt; ++idx )
	{
		_CKDWater_MatchBox box;
		box.m_iCol = iColCnt;
		box.m_iRow = idx;

		int iLastBoxIdx0 = _GetBoxByRowCol(vecBox, iMaxCol, box.m_iRow);
		int iLastBoxIdx1 = _GetBoxByRowCol(vecBox, iMaxCol - 1, box.m_iRow);
		if ((iLastBoxIdx0 >= 0) && (iLastBoxIdx1 >= 0))
		{
			_CKDWater_MatchBox* pBox0 = &(vecBox[iLastBoxIdx0]);
			_CKDWater_MatchBox* pBox1 = &(vecBox[iLastBoxIdx1]);
			box.ptCent = (pBox0->ptCent) + (pBox0->ptCent - pBox1->ptCent);

			pBox0->imgMatch.copyTo(box.imgMatch);

			_CopyImage2Box(box, srcImgGray);

			vecBox.push_back(box);
		}
	}

	
	// 补行
	for ( int idx = 0; idx < iColCnt + 1; ++idx )
	{
		_CKDWater_MatchBox box;
		box.m_iCol = idx;
		box.m_iRow = iRowCnt;

		int iLastBoxIdx0 = _GetBoxByRowCol(vecBox, box.m_iCol, iMaxRow);
		int iLastBoxIdx1 = _GetBoxByRowCol(vecBox, box.m_iCol, iMaxRow - 1);
		
		if ( (iLastBoxIdx0 >= 0) && (iLastBoxIdx1 >= 0) )
		{
			_CKDWater_MatchBox* pBox0 = &(vecBox[iLastBoxIdx0]);
			_CKDWater_MatchBox* pBox1 = &(vecBox[iLastBoxIdx1]);
			box.ptCent = (pBox0->ptCent) + (pBox0->ptCent - pBox1->ptCent);

			pBox0->imgMatch.copyTo( box.imgMatch );
			_CopyImage2Box( box, srcImgGray );
			
			vecBox.push_back( box );
		}
	}
#endif
	
}


void CAlgo_KDWater2::_SearchDefect( std::vector<_TKDWater_Defect2>& vecDefect, _CKDWater_MatchBox& box, cv::Mat& srcImgGray, int iDir )
{
	vecDefect.clear();
	
	std::vector<_CKDWater_MatchBox>  vecBox;
	vecBox.push_back(box);
	int iIndex_R = 0;
	while ( iIndex_R < vecBox.size() )
	{
		_CKDWater_MatchBox curBox = vecBox[iIndex_R];
		++iIndex_R;
		
		_SearchBox_Neib( vecBox, curBox, srcImgGray);
	}
	
	// 统一排序 Row, Col
	_UniteSortRowCol( vecBox, iDir );

	// 补齐 边缘的格子
	_CompleteRoundBox( vecBox, srcImgGray, iDir );

	// 计算亮度
	for ( int idx = 0; idx < vecBox.size( ); ++idx )
	{
		vecBox[idx].CalcMeanVal( m_tParam.iThresd_PixelValue_Water );
	}


// 	for (int idx = 0; idx < vecBox.size(); ++idx)
// 	{
// 		char buf[128];
// 		sprintf_s(buf, "D:\\1\\%d_%d_%d.bmp", idx, vecBox[idx].m_iRow, vecBox[idx].m_iCol);
// 		cv::imwrite(buf, vecBox[idx].imgMatch);
// 	}


	//----------< 计算瑕疵 >------------

	int iMaxRow = 0;
	int iMaxCol = 0;
	for (int idx = 0; idx < vecBox.size(); ++idx)
	{
		if (vecBox[idx].m_iRow > iMaxRow)
		{
			iMaxRow = vecBox[idx].m_iRow;
		}
		if (vecBox[idx].m_iCol > iMaxCol)
		{
			iMaxCol = vecBox[idx].m_iCol;
		}
	}
	
	std::vector<_TKDWater_Defect2>  vecDefect_Tmp;
	vecDefect_Tmp.reserve(64);
	for (int idx = 0; idx < vecBox.size(); ++idx)
	{
		if ( 7 == idx )
		{
			Sleep( 0 );
		}
		
		_GetDefect_Neib( vecDefect_Tmp, vecBox, idx, iDir, iMaxCol, iMaxRow, srcImgGray );
	}

	if ( vecDefect_Tmp.size() > 0 )
	{
		vecDefect.reserve( vecDefect_Tmp.size() );

		for ( int idx = 0; idx < vecDefect_Tmp.size(); ++idx )
		{
			cv::Rect roiRect = vecDefect_Tmp[idx].rtRect;
			if ( (roiRect.x >= 0)
				&& (roiRect.y >= 0)
				&& (roiRect.x + roiRect.width <= srcImgGray.cols)
				&& (roiRect.y + roiRect.height <= srcImgGray.rows) )
			{
				cv::Rect rect_Tmp = vecDefect_Tmp[idx].rtRect;
				cv::Point cent_Tmp = cv::Point(rect_Tmp.x + (rect_Tmp.width >> 1), rect_Tmp.y + (rect_Tmp.height >> 1));
				
				int total = vecDefect.size();
				bool bFlag = true;

				for ( int kk = 0; kk < total; ++kk )
				{
					cv::Rect rect_Src = vecDefect[kk].rtRect;
					cv::Point cent_Src = cv::Point(rect_Src.x + (rect_Src.width >> 1), rect_Src.y + (rect_Src.height >> 1));

					if ( ((cent_Src.x > rect_Tmp.x)
						&& (cent_Src.y > rect_Tmp.y)
						&& (cent_Src.x < rect_Tmp.x + rect_Tmp.width)
						&& (cent_Src.y < rect_Tmp.y + rect_Tmp.height))
						|| ((cent_Tmp.x > rect_Src.x)
							&& (cent_Tmp.y > rect_Src.y)
							&& (cent_Tmp.x < rect_Src.x + rect_Src.width)
							&& (cent_Tmp.y < rect_Src.y + rect_Src.height)) )
					{
						if ( vecDefect[kk].iArea < vecDefect_Tmp[idx].iArea )
						{
							vecDefect[kk] = vecDefect_Tmp[idx];
						}

						bFlag = false;
						break;
					}
				}

				if ( bFlag )
				{
					vecDefect.push_back( vecDefect_Tmp[idx] );
				}
			}
		}
	}

	//----------------------------------------------------------------

	if( 0 )
	{
		cv::Mat imgColor;
		cv::cvtColor(srcImgGray, imgColor, cv::COLOR_GRAY2BGR);
		char buf[1024];
		int iHalfW = m_imgTempl.cols >> 1;
		int iHalfH = m_imgTempl.rows >> 1;
		for ( int idx = 0; idx < vecBox.size(); ++idx )
		{
			cv::Rect rect( vecBox[idx].ptCent.x - iHalfW, vecBox[idx].ptCent.y - iHalfH, m_imgTempl.cols, m_imgTempl.rows);
			cv::rectangle( imgColor, rect, cv::Scalar(0, 255, 0), 2);
			
			sprintf_s( buf, "%d", idx);
			cv::putText( imgColor, buf, vecBox[idx].ptCent, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
			sprintf_s(buf, "%d, %d", vecBox[idx].m_iRow, vecBox[idx].m_iCol);
			cv::putText(imgColor, buf, vecBox[idx].ptCent + cv::Point(0, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);

			//sprintf_s( buf, "%d_%d.jpg", vecBox[idx].m_iRow, vecBox[idx].m_iCol);
			//cv::imwrite( buf, vecBox[idx].imgMatch );


// 			cv::Rect rect(vecBox[idx].ptCent.x - iHalfW, vecBox[idx].ptCent.y - iHalfH, m_imgTempl.cols, m_imgTempl.rows);
// 			cv::circle( imgColor, vecBox[idx].ptCent, 10, cv::Scalar(0, 255, 0), 3 );

		}

		cv::imwrite("imgColor.jpg", imgColor);
	}
}



inline int _GetBoxByRowCol_NoSpecTab( std::vector<_CKDWater_MatchBox>& vecBox, int ix, int iy )
{
	if ( ((0 == ix) && (3 == iy))
		|| ((0 == ix) && (4 == iy))
		|| ((3 == ix) && (0 == iy))
		|| ((4 == ix) && (0 == iy)) )
	{
		return -1;
	}

	int ret = -1;
	int iTotal = vecBox.size();
	for ( int idx = 0; idx < iTotal; ++idx )
	{
		if ( (vecBox[idx].m_iCol == ix) && (vecBox[idx].m_iRow == iy) )
		{
			ret = idx;
			break;
		}
	}
	return ret;
}


// 排除头，尾
inline int _GetBoxByRowCol_NotSE( std::vector<_CKDWater_MatchBox>& vecBox, int ix, int iy, int iMaxX, int iMaxY )
{
	if ( (0 == ix) || ( 0 == iy )
		|| (iMaxX == ix) || (iMaxY == iy) )
	{
		return -1;
	}

	int ret = -1;
	int iTotal = vecBox.size();
	for (int idx = 0; idx < iTotal; ++idx)
	{
		if ((vecBox[idx].m_iCol == ix) && (vecBox[idx].m_iRow == iy))
		{
			ret = idx;
			break;
		}
	}
	return ret;
}




void CAlgo_KDWater2::_GetDefect_Neib(std::vector<_TKDWater_Defect2>& vecDefect, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat& srcImgGray)
{
	int iTotal = vecBox.size();
	if (iTotal <= 0)
	{
		return;
	}

	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;
	if (5 == iCurRow) {
		int q = 1;
	}

	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;
	cv::Mat imgBoxFull = vecBox[iCurIndex].imgBoxFull;
	int bbox_x = curBox.ptCent.x - (m_imgTempl.cols >> 1);
	int bbox_y = curBox.ptCent.y - (m_imgTempl.rows >> 1);
	int bbox_w = imgCur.size().width;
	int bbox_h = imgCur.size().height;
	if ((0 == iCurRow))
	{
		bbox_h += 10;
	}
	if ((0 == iCurRow))
	{
		bbox_w += 10;
	}
	//switch (iDir)
	//{
	//case 0:
	//	bbox_x -= 10;
	//	bbox_y -= 10;
	//case 1:
	//	bbox_x -= 10;
	//case 2:
	//	bbox_y -= 10;
	//case 3:
	//	break;
	//default:
	//	break;
	//}
	//cv::Rect curRect(bbox_x, bbox_y, bbox_w, bbox_h);
	bbox_x = curBox.ptCent.x - ((m_imgTempl.cols) >> 1);
	bbox_y = curBox.ptCent.y - ((m_imgTempl.rows) >> 1);
	//bbox_w = m_imgTempl.cols + 40;
	//bbox_h = m_imgTempl.rows + 40;
	bbox_w = m_imgTempl.cols;
	bbox_h = m_imgTempl.rows;
	cv::Rect curRect(bbox_x, bbox_y, bbox_w, bbox_h);

	if (curRect.x + curRect.width >= 2448 || curRect.y + curRect.height >= 2048) {
		return;
	}	
	if (curRect.x<0 || curRect.y<0) {
		return;
	}
	/////////////////////////////////////////////////
	cv::Mat imgBoxColor;
	std::vector<cv::Mat> img_batch;
	img_color(curRect).copyTo(imgBoxColor);
	std::vector<std::vector<std::vector<cv::Point>>> contours;
	cv::imwrite("imgBoxColor.jpg", imgBoxColor);
	char buf[128];
	sprintf_s(buf, "_GetDefect_Neib \n");
	OutputDebugStringA(buf);
	std::vector<std::vector<Detection>> res_batch;
	auto& res = yoloseg->Predict(imgBoxColor, res_batch, contours);
	cv::Mat img = imgBoxColor;
	for (size_t j = 0; j < res.size(); j++) {
		// 
		_TKDWater_Defect2 defect;

		defect.iMeanVal = res[j].conf*100;

		defect.rtRect = get_rect(img, res[j].bbox);
		// 判断边缘 去除
		if (defect.rtRect.x > 238 || defect.rtRect.y > 238 || 
			defect.rtRect.x + defect.rtRect.width < 10 ||
			defect.rtRect.y + defect.rtRect.height < 10) {
			continue;
		}
		// 去除0_3角上的
		if ((0 == iCurRow && 3 == iCurCol)|| (3 == iCurRow && 0 == iCurCol) || (0 == iCurRow && 4 == iCurCol) || (4 == iCurRow && 0 == iCurCol))
		{
			if (!(
				((defect.rtRect.y + defect.rtRect.height / 2 > 35)&&(defect.rtRect.y + defect.rtRect.height / 2 < 213)) ||
				((defect.rtRect.y + defect.rtRect.height / 2 > 35) && (defect.rtRect.y + defect.rtRect.height / 2 < 213))
				)) {
				continue;
			}
		}
		defect.iArea = defect.rtRect.area();
		defect.rtRect.x += curRect.x;
		defect.rtRect.y += curRect.y;
		defect.p0.x = curRect.x;
		defect.p0.y = curRect.y;
		defect.contour = contours[j];
		vecDefect.push_back(defect);
	}

	return;

	cv::Mat imgBin = cv::Mat::zeros(imgCur.size(), CV_8UC1);

	if ((iCurCol > 0 && iCurCol < iMaxCol) && (iCurRow > 0 && iCurRow < iMaxRow))
	{
		_GetDefectImgBin_Normal(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow);
	}
	else if ((0 == iCurCol) || (iMaxCol == iCurCol))
	{
		_GetDefectImgBin_Col0(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow, srcImgGray);
	}
	else if ((0 == iCurRow) || (iMaxRow == iCurRow))
	{
		_GetDefectImgBin_Row0(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow, srcImgGray);
	}

	//cv::Mat imgBaseBin;
	//cv::threshold( imgCur, imgBaseBin, m_tParam.iThresd_PixelValue_Water, 255, cv::THRESH_BINARY_INV );
	//imgBin &= imgBaseBin;
	imgBin &= vecBox[iCurIndex].imgRegionWater;

	//------------------------------------------------------------

	_CheckSpecTab(imgBin, vecBox[iCurIndex], iDir);


#if 0

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel);

	//
	cv::Mat kernel0 = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel0);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel0);

#else

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel);

	std::vector<std::vector<cv::Point>> vecCont_t;
	cv::findContours(imgBin, vecCont_t, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	imgBin.setTo(0);
	for (int idx = 0; idx < vecCont_t.size(); ++idx)
	{
		int area = cv::contourArea(vecCont_t[idx]);
		if (area < 50)
		{
			continue;
		}
		cv::drawContours(imgBin, vecCont_t, idx, cv::Scalar::all(255), -1);
	}

	cv::Mat imgErode;
	kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgErode, cv::MorphTypes::MORPH_ERODE, kernel);

#endif

	std::vector<std::vector<cv::Point>> vecCont;
	cv::findContours(imgBin, vecCont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::Mat imgTmp = cv::Mat::zeros(imgBin.size(), CV_8UC1);

	for (int idx = 0; idx < vecCont.size(); ++idx)
	{
		int area = cv::contourArea(vecCont[idx]);
		cv::RotatedRect rotRect = cv::minAreaRect(vecCont[idx]);
		double dMinSize = rotRect.size.width < rotRect.size.height ? rotRect.size.width : rotRect.size.height;

#if 1
		if ((area < m_tParam.iThresd_BlobArea) || (dMinSize < 7))
		{
			continue;
		}

		imgTmp.setTo(0);
		cv::drawContours(imgTmp, vecCont, idx, cv::Scalar::all(255), -1);
		imgTmp &= imgErode;
		if (cv::sum(imgTmp)[0] / 255 < 4)
		{
			continue;
		}

#endif

		cv::Rect roiRect_Defect = cv::boundingRect(vecCont[idx]);

		int meanV;
		if (_CheckDefect_ByValue(meanV, imgCur, vecCont, idx))
		{
			bool bCheck = false;
			if (area > 500)
			{
				bCheck = true;
			}
			else
			{
				cv::Rect dfRect = cv::boundingRect(vecCont[idx]);
				bCheck = _CheckDefect_ByRound(imgCur, dfRect, m_tParam.iThresd_Diff - 10);
			}

			if (bCheck)
			{
				_TKDWater_Defect2 defect;

				defect.iArea = area;
				defect.iMeanVal = meanV;

				defect.rtRect = cv::boundingRect(vecCont[idx]);
				defect.rtRect.x += curRect.x;
				defect.rtRect.y += curRect.y;
				vecDefect.push_back(defect);
			}
		}

	}

}

void CAlgo_KDWater2::_GetDefect_Neib2(std::vector<_TKDWater_Defect2>& vecDefect, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat& srcImgGray)
{
	int iTotal = vecBox.size();
	if (iTotal <= 0)
	{
		return;
	}

	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;

	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;
	cv::Mat imgBin = cv::Mat::zeros(imgCur.size(), CV_8UC1);

	if ((iCurCol > 0 && iCurCol < iMaxCol) && (iCurRow > 0 && iCurRow < iMaxRow))
	{
		_GetDefectImgBin_Normal(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow);
	}
	else if ((0 == iCurCol) || (iMaxCol == iCurCol))
	{
		_GetDefectImgBin_Col0(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow, srcImgGray);
	}
	else if ((0 == iCurRow) || (iMaxRow == iCurRow))
	{
		_GetDefectImgBin_Row0(imgBin, vecBox, iCurIndex, iDir, iMaxCol, iMaxRow, srcImgGray);
	}

	//cv::Mat imgBaseBin;
	//cv::threshold( imgCur, imgBaseBin, m_tParam.iThresd_PixelValue_Water, 255, cv::THRESH_BINARY_INV );
	//imgBin &= imgBaseBin;
	imgBin &= vecBox[iCurIndex].imgRegionWater;

	//------------------------------------------------------------

	_CheckSpecTab(imgBin, vecBox[iCurIndex], iDir);


#if 0

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel);

	//
	cv::Mat kernel0 = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel0);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel0);

#else

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel);

	std::vector<std::vector<cv::Point>> vecCont_t;
	cv::findContours(imgBin, vecCont_t, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	imgBin.setTo(0);
	for (int idx = 0; idx < vecCont_t.size(); ++idx)
	{
		int area = cv::contourArea(vecCont_t[idx]);
		if (area < 50)
		{
			continue;
		}
		cv::drawContours(imgBin, vecCont_t, idx, cv::Scalar::all(255), -1);
	}

	cv::Mat imgErode;
	kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_ELLIPSE, cv::Size(7, 7));
	cv::morphologyEx(imgBin, imgErode, cv::MorphTypes::MORPH_ERODE, kernel);

#endif

	std::vector<std::vector<cv::Point>> vecCont;
	cv::findContours(imgBin, vecCont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::Rect curRect(curBox.ptCent.x - (m_imgTempl.cols >> 1), curBox.ptCent.y - (m_imgTempl.rows >> 1), m_imgTempl.cols, m_imgTempl.rows);
	cv::Mat imgTmp = cv::Mat::zeros(imgBin.size(), CV_8UC1);

	for (int idx = 0; idx < vecCont.size(); ++idx)
	{
		int area = cv::contourArea(vecCont[idx]);
		cv::RotatedRect rotRect = cv::minAreaRect(vecCont[idx]);
		double dMinSize = rotRect.size.width < rotRect.size.height ? rotRect.size.width : rotRect.size.height;

#if 1
		if ((area < m_tParam.iThresd_BlobArea) || (dMinSize < 7))
		{
			continue;
		}

		imgTmp.setTo(0);
		cv::drawContours(imgTmp, vecCont, idx, cv::Scalar::all(255), -1);
		imgTmp &= imgErode;
		if (cv::sum(imgTmp)[0] / 255 < 4)
		{
			continue;
		}

#endif

		cv::Rect roiRect_Defect = cv::boundingRect(vecCont[idx]);

		int meanV;
		if (_CheckDefect_ByValue(meanV, imgCur, vecCont, idx))
		{
			bool bCheck = false;
			if (area > 500)
			{
				bCheck = true;
			}
			else
			{
				cv::Rect dfRect = cv::boundingRect(vecCont[idx]);
				bCheck = _CheckDefect_ByRound(imgCur, dfRect, m_tParam.iThresd_Diff - 10);
			}

			if (bCheck)
			{
				_TKDWater_Defect2 defect;

				defect.iArea = area;
				defect.iMeanVal = meanV;

				defect.rtRect = cv::boundingRect(vecCont[idx]);
				defect.rtRect.x += curRect.x;
				defect.rtRect.y += curRect.y;
				defect.p0.x = curRect.x;
				defect.p0.y = curRect.y;
				defect.contour.push_back(vecCont[idx]);
				vecDefect.push_back(defect);
			}
		}

	}

}


// 中间格子
void CAlgo_KDWater2::_GetDefectImgBin_Normal( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow )
{
	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;

	int iNeibIndex[4] = { -1, -1, -1, -1 };
	std::vector<int> vecNeibIndex;

	int iIndex = _GetBoxByRowCol_NotSE(vecBox, iCurCol - 1, iCurRow, iMaxCol, iMaxRow);
	if (iIndex >= 0)
	{
		iNeibIndex[0] = iIndex;
		vecNeibIndex.push_back(iIndex);
	}

	iIndex = _GetBoxByRowCol_NotSE(vecBox, iCurCol + 1, iCurRow, iMaxCol, iMaxRow);
	if (iIndex >= 0)
	{
		iNeibIndex[1] = iIndex;
		vecNeibIndex.push_back(iIndex);
	}

	iIndex = _GetBoxByRowCol_NotSE(vecBox, iCurCol, iCurRow - 1, iMaxCol, iMaxRow);
	if (iIndex >= 0)
	{
		iNeibIndex[2] = iIndex;
		vecNeibIndex.push_back(iIndex);
	}

	iIndex = _GetBoxByRowCol_NotSE(vecBox, iCurCol, iCurRow + 1, iMaxCol, iMaxRow);
	if (iIndex >= 0)
	{
		iNeibIndex[3] = iIndex;
		vecNeibIndex.push_back(iIndex);
	}

	if ( vecNeibIndex.size() >= 3 )
	{
		_GetDefectImgBin_Neib34( imgBinOut, vecBox, iCurIndex, iDir, iNeibIndex );
	}
	else if (vecNeibIndex.size() >= 2)
	{
		_GetDefectImgBin_Neib2( imgBinOut, vecBox, iCurIndex, iDir, &(vecNeibIndex[0] ) );
	}
}



void CAlgo_KDWater2::_GetDefectImgBin_Col0( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat& srcImgGray )
{
	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;

	std::vector<int> vecNeibIndex;

	int iIndex = 0;
	if ( iCurRow <= 0 )
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow + 1 );
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}

		if (0 == iCurCol)
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol + 1, iCurRow);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
		else
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol - 1, iCurRow);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
	}
	else if ( iCurRow >= iMaxRow )
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow - 1);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}

		if ( 0 == iCurCol )
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol + 1, iCurRow);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
		else
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol - 1, iCurRow);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
	}
	else
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow - 1);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
		iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow + 1);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
	}

	if (vecNeibIndex.size() >= 2)
	{
		_GetDefectImgBin_Neib2( imgBinOut, vecBox, iCurIndex, iDir, &(vecNeibIndex[0]) );
	}

}



void CAlgo_KDWater2::_GetDefectImgBin_Row0( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow, cv::Mat& srcImgGray )
{
	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;

	std::vector<int> vecNeibIndex;

	int iIndex = 0;
	if (iCurCol <= 0)
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol + 1, iCurRow);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
		
		if (0 == iCurRow)
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow + 1);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
		else
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow - 1);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
	}
	else if (iCurCol >= iMaxCol)
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol - 1, iCurRow);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
		if (0 == iCurRow)
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow + 1);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
		else
		{
			iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow - 1);
			if (iIndex >= 0)
			{
				vecNeibIndex.push_back(iIndex);
			}
		}
	}
	else
	{
		iIndex = _GetBoxByRowCol(vecBox, iCurCol - 1, iCurRow);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
		iIndex = _GetBoxByRowCol(vecBox, iCurCol + 1, iCurRow);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
	}

	if (vecNeibIndex.size() >= 2)
	{
		_GetDefectImgBin_Neib2(imgBinOut, vecBox, iCurIndex, iDir, &(vecNeibIndex[0]));
	}
}




// 周边格子
void CAlgo_KDWater2::_GetDefectImgBin_Side( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iMaxCol, int iMaxRow )
{
#if 0
	_CKDWater_MatchBox& curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;

	std::vector<int> vecNeibIndex;

	int iIndex = 0;

	//iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow - 1);
	iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow - 1);
	if (iIndex >= 0)
	{
		vecNeibIndex.push_back(iIndex);
	}

	//if (1 != iCurCol)
	{
		//iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow + 1);
		iIndex = _GetBoxByRowCol(vecBox, iCurCol, iCurRow + 1);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
	}

	//iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol - 1, iCurRow);
	iIndex = _GetBoxByRowCol(vecBox, iCurCol - 1, iCurRow);
	if (iIndex >= 0)
	{
		vecNeibIndex.push_back(iIndex);
	}

	//if ( 1 != iCurRow )
	{
		//iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol + 1, iCurRow);
		iIndex = _GetBoxByRowCol(vecBox, iCurCol + 1, iCurRow);
		if (iIndex >= 0)
		{
			vecNeibIndex.push_back(iIndex);
		}
	}

	if (vecNeibIndex.size() >= 2)
	{
		_GetDefectImgBin_Neib_Least2( imgBinOut, vecBox, iCurIndex, iDir, vecNeibIndex );
	}

#else

#endif



}




// 
void CAlgo_KDWater2::_GetDefectImgBin_Neib34( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iNeibIndex[4] )
{
	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;
	
	cv::Mat diff_01 = cv::Mat::zeros( imgCur.size(), CV_8UC1 );
	if ( (iNeibIndex[0] >= 0) && (iNeibIndex[1] >= 0) )
	{
		// 检测内部
		cv::Mat imgDiff0 = vecBox[iNeibIndex[0]].imgMatch - imgCur;
		cv::threshold(imgDiff0, imgDiff0, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
		cv::Mat imgDiff1 = vecBox[iNeibIndex[1]].imgMatch - imgCur;
		cv::threshold(imgDiff1, imgDiff1, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
		diff_01 = imgDiff0 & imgDiff1;
		
		diff_01 &= m_imgRegion_Inner;

		// 检测外部棱边
		// 
	}
	cv::Mat diff_23 = cv::Mat::zeros( imgCur.size(), CV_8UC1 );
	if ( (iNeibIndex[2] >= 0) && (iNeibIndex[3] >= 0) )
	{
		// 检测内部
		cv::Mat imgDiff0 = vecBox[iNeibIndex[2]].imgMatch - imgCur;
		cv::threshold(imgDiff0, imgDiff0, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
		cv::Mat imgDiff1 = vecBox[iNeibIndex[3]].imgMatch - imgCur;
		cv::threshold(imgDiff1, imgDiff1, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
		diff_23 = imgDiff0 & imgDiff1;

		diff_23 &= m_imgRegion_Inner;
		
		// 检测外部棱边
		// 
	}

	imgBinOut = diff_01 | diff_23;

}



void CAlgo_KDWater2::_GetDefectImgBin_Neib2( cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, int iNeibIndex[2] )
{
	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;

	// 检测内部
	cv::Mat imgDiff0 = vecBox[iNeibIndex[0]].imgMatch - imgCur;
	cv::threshold(imgDiff0, imgDiff0, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
	cv::Mat imgDiff1 = vecBox[iNeibIndex[1]].imgMatch - imgCur;
	cv::threshold(imgDiff1, imgDiff1, m_tParam.iThresd_Diff, 255, cv::THRESH_BINARY);
	imgBinOut = imgDiff0 & imgDiff1;

	imgBinOut &= m_imgRegion_Inner;

	// 检测外部棱边

}


void CAlgo_KDWater2::_GetDefectImgBin_Neib_Least2(cv::Mat& imgBinOut, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir, std::vector<int>& vecIndex )
{
	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;
	cv::Mat imgAcc = cv::Mat::zeros( imgCur.size(), CV_8UC1 );

	// 检测内部
	int iTotal = vecIndex.size();
	for (int idx = 0; idx < iTotal; ++idx)
	{
		cv::Mat imgDiff = vecBox[ vecIndex[idx] ].imgMatch - imgCur;
		cv::threshold( imgDiff, imgDiff, m_tParam.iThresd_Diff, 1, cv::THRESH_BINARY );

		imgAcc += imgDiff;
	}
	cv::threshold(imgAcc, imgBinOut, 1, 255, cv::THRESH_BINARY);
	
	imgBinOut &= m_imgRegion_Inner;

	// 检测外部棱边


}









void CAlgo_KDWater2::_CheckSpecTab(cv::Mat& imgBin, _CKDWater_MatchBox& curBox, int iDir)
{
	cv::Mat imgtabMask;

	// 0 - X, 1 - Y
	switch (iDir)
	{
	case 0: // LT 
		if ((3 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{
			imgtabMask = m_imgTempl_tabMask_Col;
			imgBin &= imgtabMask;
		}
		else if ((4 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, 1);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (3 == curBox.m_iRow))
		{
			imgtabMask = m_imgTempl_tabMask_Row;
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (4 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, 0);
			imgBin &= imgtabMask;
		}
		break;

	case 1:  // LB
		if ((3 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, 0);
			imgBin &= imgtabMask;
		}
		else if ((4 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, -1);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (3 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, 0);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (4 == curBox.m_iRow))
		{
			imgtabMask = m_imgTempl_tabMask_Row;
			imgBin &= imgtabMask;
		}
		break;

	case 2: // RT
		if ((3 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{  // 行
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, 1);
			imgBin &= imgtabMask;
		}
		else if ((4 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{  // 行
			imgtabMask = m_imgTempl_tabMask_Col;
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (3 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, 1);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (4 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, -1);
			imgBin &= imgtabMask;
		}
		break;

	default:  // RB
		if ((3 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{  // 行
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, 1);
			cv::flip(imgtabMask, imgtabMask, 0);
			imgBin &= imgtabMask;
		}
		else if ((4 == curBox.m_iCol) && (0 == curBox.m_iRow))
		{  // 行
			cv::flip(m_imgTempl_tabMask_Col, imgtabMask, 0);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (3 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, -1);
			imgBin &= imgtabMask;
		}
		else if ((0 == curBox.m_iCol) && (4 == curBox.m_iRow))
		{
			cv::flip(m_imgTempl_tabMask_Row, imgtabMask, 1);
			imgBin &= imgtabMask;
		}
		break;
	}
}


// 平均亮度
bool CAlgo_KDWater2::_CheckDefect_ByValue( int& iMeanVOut, cv::Mat& imgCur, std::vector<std::vector<cv::Point>>& vecCont, int iContIdx )
{
	cv::Mat imgTmp = cv::Mat::zeros(imgCur.size(), CV_8UC1 );
	cv::drawContours( imgTmp, vecCont, iContIdx, cv::Scalar::all(255), -1 );
	int meanV = cv::mean(imgCur, imgTmp)[0];
	iMeanVOut = meanV;

// 	cv::Rect rect = cv::boundingRect(vecCont[iContIdx]);
// 	int meanV_Rect = cv::mean(imgCur(rect))[0];
// 

	bool ret = false;
	if ( meanV < m_tParam.iThresd_MaxMeanValue )
	{
		ret = true;
	}
	
	return ret;
}


bool CAlgo_KDWater2::_CheckDefect_ByRound( cv::Mat& imgCur, cv::Rect &rtDefect, int iThresd_NeibDiff )
{
	int iMeanV_Std = cv::mean(imgCur(rtDefect))[0];
	
	cv::Point dfCent;
	dfCent.x = rtDefect.x + (rtDefect.width >> 1);
	dfCent.y = rtDefect.y + (rtDefect.height >> 1);

	cv::Point imgCent;
	imgCent.x = imgCur.cols >> 1;
	imgCent.y = imgCur.rows >> 1;


	if ( (abs( dfCent.x - imgCent.x ) < 40)
		&& (abs(dfCent.y - imgCent.y) < 40) )
	{
		return true;
	}

	cv::Rect rtTmp = rtDefect;
	if ( dfCent.x < imgCent.x )
	{
		rtTmp.x += rtDefect.width;
	}
	else
	{
		rtTmp.x -= rtDefect.width;
	}
	if (dfCent.y < imgCent.y)
	{
		rtTmp.y += rtDefect.height;
	}
	else
	{
		rtTmp.y -= rtDefect.height;
	}

	if ( rtTmp.x < 0 
		|| rtTmp.x + rtTmp.width > imgCur.cols
		|| rtTmp.y < 0 
		|| rtTmp.y + rtTmp.height > imgCur.rows )
	{
		return false;
	}

	int iMeanV_Tmp = cv::mean(imgCur(rtTmp))[0];
	
	return iMeanV_Std + iThresd_NeibDiff < iMeanV_Tmp ? true : false;



#if 0
	int iMeanV_Std = cv::mean(imgCur(rtDefect))[0];

	cv::Rect rtTmp = rtDefect;
	rtTmp.x -= rtDefect.width + 10;
	if ( rtTmp.x > 0 )
	{
		int iMeanV_Tmp = cv::mean(imgCur(rtTmp))[0];
		if (iMeanV_Std + iThresd_NeibDiff < iMeanV_Tmp)
		{
			return true;
		}
	}

	rtTmp = rtDefect;
	rtTmp.x += rtDefect.width + 10;
	if (rtTmp.x + rtTmp.width < imgCur.cols )
	{
		int iMeanV_Tmp = cv::mean(imgCur(rtTmp))[0];
		if (iMeanV_Std + iThresd_NeibDiff < iMeanV_Tmp)
		{
			return true;
		}
	}

	rtTmp = rtDefect;
	rtTmp.y -= rtDefect.height + 10;
	if (rtTmp.y > 0)
	{
		int iMeanV_Tmp = cv::mean(imgCur(rtTmp))[0];
		if (iMeanV_Std + iThresd_NeibDiff < iMeanV_Tmp)
		{
			return true;
		}
	}

	rtTmp = rtDefect;
	rtTmp.y += rtDefect.height + 10;
	if (rtTmp.y + rtTmp.height < imgCur.rows )
	{
		int iMeanV_Tmp = cv::mean(imgCur(rtTmp))[0];
		if (iMeanV_Std + iThresd_NeibDiff < iMeanV_Tmp)
		{
			return true;
		}
	}

	return false;
#endif
}


void CAlgo_KDWater2::_SearchBox_Neib( std::vector<_CKDWater_MatchBox>& vecBox, _CKDWater_MatchBox& curBox, cv::Mat& srcImgGray )
{
	_GetNeibBox( curBox, cv::Point(-m_ptStdOffs.x, 0), vecBox, srcImgGray );  // Left
	_GetNeibBox( curBox, cv::Point(m_ptStdOffs.x, 0), vecBox, srcImgGray );   // Right
	_GetNeibBox( curBox, cv::Point(0, -m_ptStdOffs.y), vecBox, srcImgGray );  // Up
	_GetNeibBox( curBox, cv::Point(0, m_ptStdOffs.y), vecBox, srcImgGray );   // Down
}



bool CAlgo_KDWater2::_GetNeibBox( _CKDWater_MatchBox& curBox, cv::Point offsPt, std::vector<_CKDWater_MatchBox>& vecBox, cv::Mat& srcImgGray )
{
	int iIndex = -1;
	cv::Point  dstCent = curBox.ptCent + offsPt;
	for (int idx = 0; idx < vecBox.size(); ++idx)
	{
		if (vecBox[idx].IsPtInRect(dstCent))
		{
			iIndex = idx;
			break;
		}
	}

	if (iIndex >= 0)
	{
		return true;
	}

	//////////////////////////////////////////////////////////

	int iMatchBoxHalfWid = (m_imgTempl.cols >> 1) + 30;
	int iMatchBoxHalfHei = (m_imgTempl.rows >> 1) + 30;
	int iMatchBoxWid = iMatchBoxHalfWid << 1;
	int iMatchBoxHei = iMatchBoxHalfHei << 1;
	bool ret = false;

	cv::Rect srcRect(dstCent.x - iMatchBoxHalfWid, dstCent.y - iMatchBoxHalfHei, iMatchBoxWid, iMatchBoxHei);

	_CKDWater_MatchBox outBox;

	if ((srcRect.x >= 0)
		&& (srcRect.y >= 0)
		&& (srcRect.x + srcRect.width < srcImgGray.cols)
		&& (srcRect.y + srcRect.height < srcImgGray.rows))
	{
		cv::Mat matchImg;
		srcImgGray(srcRect).copyTo(matchImg);
		if (true == _GetMaxMatch(outBox, matchImg, curBox.imgMatch))
		{
			outBox.ptCent.x += srcRect.x;
			outBox.ptCent.y += srcRect.y;
			ret = true;
		}
	}

	if ( true == ret )
	{
		outBox.m_iCol = curBox.m_iCol + (offsPt.x > 0 ? 1 : ((offsPt.x < 0 ? -1 : 0)));
		outBox.m_iRow = curBox.m_iRow + (offsPt.y > 0 ? 1 : ((offsPt.y < 0 ? -1 : 0)));
		vecBox.push_back( outBox );
	}

	return ret;
}



bool CAlgo_KDWater2::_GetMaxMatch( _CKDWater_MatchBox& outBox, cv::Mat& roiImg, cv::Mat& imgTempl )
{
	cv::Mat imgTempl_Sub;
	cv::pyrDown(imgTempl, imgTempl_Sub);
	cv::Mat imgRoi_Sub;
	cv::pyrDown(roiImg, imgRoi_Sub);

	cv::Mat result;
	cv::matchTemplate(imgRoi_Sub, imgTempl_Sub, result, cv::TemplateMatchModes::TM_CCOEFF_NORMED);
	result.convertTo(result, CV_8U, 255);
	double dMax = 0;
	cv::minMaxIdx(result, NULL, &dMax);

	int iVal = dMax;
	if (iVal < 180)
	{
		return false;
	}

	int iSumX = 0;
	int iSumY = 0;
	int cnt = 0;
	for (int row = 0; row < result.rows; ++row)
	{
		uchar* pLine = result.ptr<uchar>(row);
		for (int col = 0; col < result.cols; ++col)
		{
			if (pLine[col] >= iVal)
			{
				iSumX += col;
				iSumY += row;
				++cnt;
			}
		}
	}
	if (cnt <= 0)
	{
		return false;
	}

	outBox.ptCent.x = iSumX * 2 / cnt;
	outBox.ptCent.y = iSumY * 2 / cnt;

	cv::Rect roiRect(outBox.ptCent.x, outBox.ptCent.y, imgTempl.cols, imgTempl.rows);
	roiImg(roiRect).copyTo(outBox.imgMatch);

	outBox.ptCent.x += imgTempl.cols >> 1;
	outBox.ptCent.y += imgTempl.rows >> 1;

	return true;
}



bool CAlgo_KDWater2::_GetBestMatchPos( cv::Point& ptCent, cv::Mat& srcImgGray )
{
	cv::Mat result;
	cv::matchTemplate( srcImgGray, m_imgTempl, result, cv::TemplateMatchModes::TM_CCOEFF_NORMED );

	result.convertTo(result, CV_8U, 255);

	double dMax = 0;
	cv::minMaxIdx(result, NULL, &dMax);

	dMax -= 1.0f;
	if ( dMax < 100 )
	{
		dMax = 100;
	}

	cv::Mat imgBin;
	cv::threshold(result, imgBin, dMax, 255, cv::THRESH_BINARY);

	std::vector<std::vector<cv::Point>> vecCont;
	cv::findContours(imgBin, vecCont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	std::vector<cv::Rect> vecRect;
	int iMaxV = 0;
	int iMaxIdx = -1;
	for (int idx = 0; idx < vecCont.size(); ++idx)
	{
		int area = vecCont[idx].size();
		if (area > iMaxV)
		{
			iMaxV = area;
			iMaxIdx = idx;
		}
	}

	bool ret = false;
	if ( iMaxIdx >= 0 )
	{
		cv::Rect rect = cv::boundingRect( vecCont[iMaxIdx] );

		ptCent.x = rect.x + ( rect.width >> 1 ) + ( m_imgTempl.cols >> 1 );
		ptCent.y = rect.y + ( rect.height >> 1 ) + ( m_imgTempl.rows >> 1 );

		ret = true;
	}

	return ret;
}









#if 0
void CAlgo_KDWater2::_GetDefect_Neib(std::vector<_TKDWater_Defect2>& vecDefect, std::vector<_CKDWater_MatchBox>& vecBox, int iCurIndex, int iDir)
{
	if (vecBox.size() < 2)
	{
		return;
	}

	bool bNeib[4] = { false, false, false, false };
	int iNeibIndex[4] = { iCurIndex,iCurIndex,iCurIndex,iCurIndex };
	int  iCntValid = 0;

	_CKDWater_MatchBox curBox = vecBox[iCurIndex];
	int iCurRow = curBox.m_iRow;
	int iCurCol = curBox.m_iCol;
	int iTotal = vecBox.size();

	int iIndex = 0;

	if (0 != iCurCol)
	{
		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol - 1, iCurRow);
		if (iIndex >= 0)
		{
			bNeib[0] = true;
			iNeibIndex[0] = iIndex;
			iCntValid++;
		}

		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol + 1, iCurRow);
		if (iIndex >= 0)
		{
			bNeib[1] = true;
			iNeibIndex[1] = iIndex;
			iCntValid++;
		}
	}

	if (0 != iCurRow)
	{
		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow - 1);
		if (iIndex >= 0)
		{
			bNeib[2] = true;
			iNeibIndex[2] = iIndex;
			iCntValid++;
		}

		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow + 1);
		if (iIndex >= 0)
		{
			bNeib[3] = true;
			iNeibIndex[3] = iIndex;
			iCntValid++;
		}
	}

	if ((0 == iCurCol) && (0 == iCurRow))
	{
		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol + 1, iCurRow);
		if (iIndex >= 0)
		{
			bNeib[1] = true;
			iNeibIndex[1] = iIndex;
			iCntValid++;
		}

		iIndex = _GetBoxByRowCol_NoSpecTab(vecBox, iCurCol, iCurRow + 1);
		if (iIndex >= 0)
		{
			bNeib[3] = true;
			iNeibIndex[3] = iIndex;
			iCntValid++;
		}
	}

	//------------------------------------------

	cv::Mat imgCur = vecBox[iCurIndex].imgMatch;
	// 180 60 50 35

	// #define DEF_THRESD_AREA    180   // 
	// #define DEF_THRESD_DIFF     60    //  -- 水渍与其他区域的对比最小值
	// #define DEF_THRESD_NOR      50    // -- 水渍区域亮度最高值
	// #define DEF_THRESD_MAXMEAN  35    //  水渍区域 的 最高平均值  


#define DEF_THRESD_AREA    150   // 
#define DEF_THRESD_DIFF     50    //  -- 水渍与其他区域的对比最小值
#define DEF_THRESD_NOR      60    // -- 水渍区域亮度最高值
#define DEF_THRESD_MAXMEAN  50    //  水渍区域 的 最高平均值  


	cv::Mat diff[4];
	cv::Mat diff_Src[4];
	cv::Mat diff_Bin[4];
	for (int idx = 0; idx < 4; ++idx)
	{
		diff[idx] = vecBox[iNeibIndex[idx]].imgMatch - imgCur;
		diff_Src[idx] = vecBox[iNeibIndex[idx]].imgMatch - imgCur;
	}

	for (int idx = 0; idx < 4; ++idx)
	{
		cv::threshold(diff[idx], diff[idx], DEF_THRESD_DIFF, 1, cv::THRESH_BINARY);
		cv::threshold(diff_Src[idx], diff_Bin[idx], DEF_THRESD_DIFF, 255, cv::THRESH_BINARY);
	}

	for (int idx = 1; idx < 4; ++idx)
	{
		diff[0] += diff[idx];
	}

	cv::Mat imgBin;
	cv::threshold(diff[0], imgBin, 1, 255, cv::THRESH_BINARY);

	cv::Mat imgBaseBin;
	cv::threshold(imgCur, imgBaseBin, DEF_THRESD_NOR, 255, cv::THRESH_BINARY_INV);
	imgBin &= imgBaseBin;

	//------------------------------------------------------------

	_CheckSpecTab(imgBin, vecBox[iCurIndex], iDir);

	//------------------------------------------------------------

	cv::Mat kernel0 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel0);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel0);

	cv::Mat kernel = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(5, 5));
	// cv::morphologyEx( imgBin, imgBin, cv::MorphTypes::MORPH_OPEN, kernel );
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_ERODE, kernel);
	cv::morphologyEx(imgBin, imgBin, cv::MorphTypes::MORPH_DILATE, kernel);

	std::vector<std::vector<cv::Point>> vecCont;
	cv::findContours(imgBin, vecCont, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	cv::Rect curRect(curBox.ptCent.x - (m_imgTempl.cols >> 1), curBox.ptCent.y - (m_imgTempl.rows >> 1), m_imgTempl.cols, m_imgTempl.rows);
	for (int idx = 0; idx < vecCont.size(); ++idx)
	{
		int area = cv::contourArea(vecCont[idx]);
		cv::RotatedRect rotRect = cv::minAreaRect(vecCont[idx]);
		double dMinSize = rotRect.size.width < rotRect.size.height ? rotRect.size.width : rotRect.size.height;

#if 0
		// Calculate the moments of the contour
		cv::Moments moments = cv::moments(vecCont[idx]);

		// Calculate the center of the contour
		cv::Point center(moments.m10 / moments.m00, moments.m01 / moments.m00);
		int iPos = 0;	// 0:中间 1:边角  2: 棱边
		if ((center.x < 30) || (center.y < 30) ||
			(center.x > 220) || (center.y > 220)) {
			iPos = 2;
		}
#endif

		if ((area > DEF_THRESD_AREA) && (dMinSize >= 7))	//-----------------------
		{
			cv::Rect roiRect_Defect = cv::boundingRect(vecCont[idx]);

			cv::Mat imgTmp = cv::Mat::zeros(imgCur.size(), CV_8UC1);
			cv::drawContours(imgTmp, vecCont, idx, cv::Scalar::all(255), -1);
			int meanV = cv::mean(imgCur, imgTmp)[0];

			if (meanV < 30)
			{
				_TKDWater_Defect2 defect;

				defect.iArea = area;
				defect.rtRect = cv::boundingRect(vecCont[idx]);

				defect.rtRect.x += curRect.x;
				defect.rtRect.y += curRect.y;
				vecDefect.push_back(defect);
			}
			else
			{
				if (meanV < DEF_THRESD_MAXMEAN)
				{
					if (_CheckDefect_ByRound(imgCur, roiRect_Defect, 40))
					{
						_TKDWater_Defect2 defect;

						defect.iArea = area;
						defect.rtRect = cv::boundingRect(vecCont[idx]);

						defect.rtRect.x += curRect.x;
						defect.rtRect.y += curRect.y;
						vecDefect.push_back(defect);
					}
				}
			}
		}
	}

}
#endif

