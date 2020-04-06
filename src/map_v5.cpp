//#include "stdafx.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>

#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv/cv.h>

#include "DfD.h"
#include "support_functions.h"

// GC
#include "gco-v3.0/GCoptimization.h"

using namespace std;
using namespace cv;

///////////////////////////////////////////////////////////////////////////////////
//																				 //
//    Part 2:																	 //
//			Subfunction of calculating logpost                                   //
//                                                                               //
///////////////////////////////////////////////////////////////////////////////////

void diffSum(int classes, std::vector<cv::Mat> &diff_Y, std::vector<cv::Mat> &diff_Cr, std::vector<cv::Mat> &diff_Cb, std::vector<cv::Mat> &diffSumMat)
{
	int kdx;
	//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	// kernel to do the addition
	cv::Mat element = cv::Mat(Size(3, 3), CV_32FC1, cv::Scalar::all(0));
	Size imgSize = diff_Y[0].size();
	cv::Mat diff_Mat = cv::Mat(imgSize, CV_32FC1, cv::Scalar::all(0));

	element.at<float>(1, 1) = 20;
	//element.at<short>(0, 1) = -2;
	//element.at<short>(1, 0) = -2;
	//element.at<short>(1, 2) = -2;
	//element.at<short>(2, 1) = -2;

	//element.at<short>(0, 0) = 0;
	//element.at<short>(0, 4) = 0;
	//element.at<short>(4, 0) = 0;
	//element.at<short>(4, 4) = 0;
	double para = 0.001;

	//element = element*(1.0 / sum(element)[0]);

	for (kdx = 0; kdx < classes; ++kdx)
	{
		// use this one for a single grayscale channel
		//filter2D(diff_Y[kdx], diffSumMat[kdx], CV_64F, element, Point(-1, -1), 0.0, BORDER_CONSTANT);

		//GaussianBlur(diff_Y[kdx], diff_Mat, Size(0, 0), para, para, BORDER_CONSTANT);
		//diff_Y[kdx].convertTo(diff_Y[kdx], CV_32FC1, 1, 0);
		//medianBlur(diff_Y[kdx], diff_Mat, 3);
		//diff_Mat.convertTo(diff_Mat, CV_64FC1, 1, 0);
		//dilate(diff_Y[kdx], diff_Mat, element);

	
		
		//GaussianBlur(diff_Cr[kdx], diff_Mat, Size(0, 0), para, para, BORDER_CONSTANT);
		//diff_Cr[kdx].convertTo(diff_Cr[kdx], CV_32FC1, 1, 0);
		//medianBlur(diff_Cr[kdx], diff_Mat, 3);
		//diff_Mat.convertTo(diff_Mat, CV_64FC1, 1, 0);
		//dilate(diff_Cr[kdx], diff_Mat, element);

		
		
		//GaussianBlur(diff_Cb[kdx], diff_Mat, Size(0, 0), para, para, BORDER_CONSTANT);
		//diff_Cb[kdx].convertTo(diff_Cb[kdx], CV_32FC1, 1, 0);
		//medianBlur(diff_Cb[kdx], diff_Mat, 3);
		//diff_Mat.convertTo(diff_Mat, CV_64FC1, 1, 0);
		//dilate(diff_Cb[kdx], diff_Mat, element);


		// original working version of diffsum
		//filter2D(diff_Y[kdx], diff_Mat, CV_64F, element, Point(-1, -1), 0.0, BORDER_CONSTANT);
		//diffSumMat[kdx] = diffSumMat[kdx] + diff_Mat;
		//filter2D(diff_Cr[kdx], diff_Mat, CV_64F, element, Point(-1, -1), 0.0, BORDER_CONSTANT);
		//diffSumMat[kdx] = diffSumMat[kdx] + diff_Mat;
		//filter2D(diff_Cb[kdx], diff_Mat, CV_64F, element, Point(-1, -1), 0.0, BORDER_CONSTANT);
		//diffSumMat[kdx] = diffSumMat[kdx] + diff_Mat;

		diffSumMat[kdx] = 12*(diff_Y[kdx] + diff_Cr[kdx] + diff_Cb[kdx]);

		// See if gray only works
		//diffSumMat[kdx] = 16 * (diff_Y[kdx]);


	}	// end of kdx loop

}	// end of diffSum

///////////////////////////////////////////////////////////////////////////////////
//    Part 3:																	 //
//			Revised MAP Estimation Function                                      //
///////////////////////////////////////////////////////////////////////////////////
//#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

void map3(string &DataLog, cv::Mat &Depth_Map, std::vector<cv::Mat> &diff_Y, std::vector<cv::Mat> &diff_Cr, std::vector<cv::Mat> &diff_Cb, int classes)
{
	int idx;
	double tick, tock, delta_T;
	double tick_Freq = ((double)cv::getTickFrequency()*1000.0);

// #if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
// #else
	// GPS_Thread_Vars *GPS_Ctrl_Info = (GPS_Thread_Vars *)(args);

	// DFD_Thread_Vars *DFD_info = (DFD_Thread_Vars *)(args);
	
	// string DataLog = DFD_info->DataLog;
	// cv::Mat Depth_Map = DFD_info->Depth_Map;
	// vector<Mat> diff_Y = DFD_info->diff_Y;
	// vector<Mat> diff_Cr = DFD_info->diff_Cr;
	// vector<Mat> diff_Cb = DFD_info->diff_Cb;
	// int classes = DFD_info->classes;
	
	
// #endif
	
	int cols = Depth_Map.cols;
	int rows = Depth_Map.rows;

	Size logSize = Depth_Map.size();
	std::vector<cv::Mat> diffSumMat;

	tick = (double)cv::getTickCount();

	for (idx = 0; idx < classes; ++idx)
	{
		diffSumMat.push_back(cv::Mat(cv::Size(cols, rows), CV_64FC1, cv::Scalar(0.0)));
	}

	diffSum(classes, diff_Y, diff_Cr, diff_Cb, diffSumMat);

	cv::Mat gridResult = cv::Mat(cv::Size(cols, rows), CV_8U, cv::Scalar(0));

	cout << "Starting GridGraph routines..." << endl;

	GridGraph_DArraySArray(cols, rows, classes, diffSumMat, gridResult, DataLog);
	//GeneralGraph_DArraySArray(cols,rows,MAX_CLASSES,logpost1,result);
		
	for (idx = 0; idx < classes; ++idx)
	{
		diffSumMat[idx].~Mat();
	}

	cv::bitwise_not(gridResult, Depth_Map);
	Depth_Map = Depth_Map + 1;

	tock = (double)cv::getTickCount();  
	delta_T = (tock - tick) / tick_Freq; 
	cout << "Elapsed time for GridGraph routine: " << delta_T << "ms." << endl;

}	// end of map3




void gen_smooth_terms(int32_t classes, int32_t smooth[])
{
    int32_t idx, jdx;
    
    //// next set up the array for smooth costs
    // @mem(smooth, INT32, 1, classes, classes, classes*4)
    for (idx = 0; idx < classes; ++idx)
    {
        for (jdx = 0; jdx < classes; ++jdx)
        {
            // orginal smoothness calculation recommended by the authors of the code <- is a metric, but doesn't work
            //smooth[idx + jdx*classes] = (idx - jdx)*(idx - jdx) <= 4 ? (idx - jdx)*(idx - jdx) : 4;

            // Chaos version of the smoothness term <- this works
            //smooth[idx + jdx*classes] = 3*std::abs(idx - jdx);

            // yes with slightly better MSE and SSIM 
            //-------------------------------------------
            smooth[idx + jdx*classes] = (int)(5.28 * 255 * std::abs(1 - std::exp(0.0015*(idx - jdx))) / (1 + std::exp(0.0015*(idx - jdx))));
            //-------------------------------------------


            // some other ideas for smoothness 
            // similar to chao's, but with a steaper climb and fast leveling off - gull wing
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 255)*(idx - 255) - (jdx - 255)*(jdx - 255)));		// yes but slow		
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 256)*(jdx + 256) - (jdx - 256)*(idx + 256)));		// yes but slow

            //smooth[idx + jdx*classes] = abs((idx - jdx)*(idx - jdx));	// no

            // semi circle <- not a "metric" error
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(idx + 127) + (jdx - 127)*(jdx + 127)));		// no

            // offset lines
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(idx + 127) - (jdx - 127)*(jdx + 127)));		// yes but slow
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 256)*(idx + 256) - (jdx - 256)*(jdx + 256)));		// yes but slow

            // not a metric - point cloud
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(idx - 127) + (jdx - 127)*(jdx - 127)));		// no

            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(jdx - 127)+255));	// no

            // not a metric <- plus sign
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(jdx - 127) + (jdx - 127)*(idx - 127)));	
            //smooth[idx + jdx*classes] = sqrt(abs((idx - 127)*(jdx + 127) - (jdx - 127)*(idx + 127)));	

            // no
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(idx - jdx) / 255)));
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(idx - 127)*(jdx-127) / (255*255))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(idx - 255)*(jdx - 255) / (255 * 255))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(255-idx)*(255-jdx) / (255 * 255))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(jdx - idx)/ (2 * 255))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(cos(2 * PI*(jdx + idx) / (2 * 255))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos(2 * PI* idx / 255)) - abs(cos(2 * PI* jdx / 255))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos(2 * PI* (idx-64) / 255)) - abs(cos(2 * PI* (jdx-64) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 64) / 255)) + abs(cos(2 * PI* (jdx - 64) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 32) / 255)) + abs(cos(2 * PI* (jdx - 32) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 16) / 255)) + abs(cos(2 * PI* (jdx - 16) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 16) / 255)) + abs(cos(2 * PI* (jdx - 16) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 16) / 255)) + abs(cos(2 * PI* (jdx - 16) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx - 16) / 255)) - abs(cos(2 * PI* (jdx - 16) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx) / 255)) - abs(sin(2 * PI* (jdx) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx) / 255)) + abs(sin(2 * PI* (jdx) / 255))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos(2 * PI* (idx) / 255)) + abs(sin(2 * PI* (jdx) / 255))));

            // metric but didn't finish in 8 hours
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos(2 * PI* (idx) / 255))));

            //no
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos(8 * PI* (128 - idx)*(128 - jdx) / (128*128) ))));

            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos((8 * PI* (255 - idx)+(255 - jdx)) / (255+255) ))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos((8 * PI* (64 - idx) + (64 - jdx)) / ( 255-64)))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos((8 * PI* (64 - idx) - (64 - jdx)) / (255 - 64)))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos((8 * PI* (idx - 64)) / (255-64) )) + abs(cos((8 * PI* (jdx - 64)) / (255-64) ))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos((8 * PI* (idx - 64)) / (255 - 64))) - abs(cos((8 * PI* (jdx - 64)) / (255 - 64)))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos((8 * PI* (idx - 64)) / (255 - 64))) - abs(sin((8 * PI* (jdx - 64)) / (255 - 64)))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos((8 * PI* (idx - 64)) / (255 - 64))) + abs(sin((8 * PI* (jdx - 64)) / (255 - 64)))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(cos((8 * PI* (64-idx)) / (255 - 64))) + abs(sin((8 * PI* (64-jdx)) / (255 - 64)))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos((8 * PI* (32+idx - jdx)) / (32+255))) ));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(cos((8 * PI* (idx - jdx)) / (255)))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(sin((8 * PI* (idx - jdx)) / (255)))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(sin((4 * PI* (idx - jdx)) / (255)))));
            //smooth[idx + jdx*classes] = (int)(255 * (abs(sin((2 * PI* (idx - jdx)) / (255)))));
            //smooth[idx + jdx*classes] = (int)(127 * (abs(sin((PI* (idx - jdx)) / (2*255)))));

            // worked but not very well, very blocky
            //smooth[idx + jdx*classes] = (int)(255 * sqrt(abs((idx - jdx) )));


            // works, but slow and splotchy
            //smooth[idx + jdx*classes] = (int)(2 * sqrt(abs((idx - jdx))));

            // slow but better
            //smooth[idx + jdx*classes] = (int)(16 * sqrt(abs((idx - jdx))));

            //no
            //smooth[idx + jdx*classes] = (int)(132*(exp(abs(idx - jdx)/255.0)-1));
            //smooth[idx + jdx*classes] = (int)(255 * (exp(abs((idx - jdx)*(idx-jdx)) / (255.0*255.0)) - 1));
            //smooth[idx + jdx*classes] = (int)(255 * (exp(abs((idx - jdx)) / (255.0*255.0)) - 1));

            // works but blocky
            //smooth[idx + jdx*classes] = (int)(255 * abs((exp(idx - jdx) - exp(jdx - idx)) / (exp(idx - jdx) + exp(jdx - idx))));

            // no
            //smooth[idx + jdx*classes] = (int)(255 * abs(1 - exp(0.1*(idx - jdx)) ) / ( 1 + exp(0.1*(idx - jdx)) ) );
            //smooth[idx + jdx*classes] = (int)(255 * abs(1 - exp(0.02*(idx - jdx))) / (1 + exp(0.02*(idx - jdx))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(1 - exp(0.004*(idx - jdx))) / (1 + exp(0.004*(idx - jdx))));
            //smooth[idx + jdx*classes] = (int)(255 * abs(1 - exp(0.0025*(idx - jdx))) / (1 + exp(0.0025*(idx - jdx))));

            // no
            //smooth[idx + jdx*classes] = (int)(5.3 * 255 * abs(1 - exp(0.0015*(idx - jdx))) / (1 + exp(0.0015*(idx - jdx))));
            //smooth[idx + jdx*classes] = (int)(256 * (abs(sin(( PI* (idx - jdx)) / (512)))));
            //a = 0.0012;
            //smooth[idx + jdx*classes] = (int)(860 * abs(exp(a*(idx - jdx)) - exp(-a*(idx - jdx))) / (exp(a*(idx - jdx)) + exp(-a*(idx - jdx))));


            // this is where the testing is going
            //smooth[idx + jdx*classes] = (int)(classes * std::sqrt(std::abs((idx - jdx)/(double)classes)));
            //smooth[idx + jdx*classes] = (int)(classes * (std::abs(idx - jdx) / ((double)classes / 2.0))*((idx - jdx) / (double)classes)*((idx - jdx) / (double)classes));
            //smooth[idx + jdx*classes] = (int)(classes * 1.0*((idx - jdx) / (double)classes)*((idx - jdx) / (double)classes));


        }
    }
}   // end of gen_smooth_terms


////////////////////////////////////////////////////////////////////////////////
// in this version, set data and smoothness terms using arrays
// grid neighborhood structure is assumed
//
//void GridGraph_DArraySArray(int width, int height, int num_labels, double **logpost1[], int *result)

void GridGraph_DArraySArray(int cols, int rows, int classes, std::vector<cv::Mat> &diffSum, cv::Mat &gridResult, std::string &DataLog)
{
	int idx, jdx, kdx;
	int num_pixels = cols * rows;
	int *result = new int[num_pixels];   // stores result of optimization

	// first set up the array for data costs
	int *data = new int[num_pixels*classes];
    int *smooth = new int[classes*classes];

	int t = 0;
	for (idx = 0; idx < rows; ++idx)
	{
		for (jdx = 0; jdx < cols; ++jdx)
		{
			for (kdx = 0; kdx < classes; ++kdx)
			{
				//data[t] = logpost1[kdx][idx][jdx]; 
				data[t] = (int)diffSum[kdx].at<double>(idx, jdx);
				++t;
			}
		}
	}

    gen_smooth_terms(classes, smooth);

	try
	{
		GCoptimizationGridGraph *gc = new GCoptimizationGridGraph(cols, rows, classes);

		gc->setDataCost(data);

		gc->setSmoothCost(smooth);
		
		std::string energy = to_string(gc->compute_energy());
		std::cout << endl << "Before optimization energy is " << energy << std::endl;
		DataLog = "\nBefore optimization energy is " + energy + "\n";

		gc->expansion();// run expansion for 2 iterations. For swap use gc->swap(num_iterations);
		
		energy = to_string(gc->compute_energy());
		std::cout << "After optimization energy is " << energy << std::endl;
		DataLog += "After optimization energy is " + energy + "\n\n";

		for (idx = 0; idx < num_pixels; ++idx)
		{
			result[idx] = gc->whatLabel(idx);
		}

		gridResult = cv::Mat(cv::Size(cols, rows), CV_32S, result, (cols * 4));
		gridResult.convertTo(gridResult, CV_8U, 1.0, 0.0);

		delete gc;
	}
	catch (GCException e)
	{
		e.Report();
	}

	delete[] smooth;
	delete[] data;
	delete[] result;

}	// end of GridGraph_DArraySArray

