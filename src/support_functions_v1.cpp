#include <cstdlib>
#include <cstdio>

#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv/cv.h>

#include <algorithm>
#include <vector> 
#include <iostream>
#include <sstream>
#include <fstream>
#include <ctime>

#include "DfD.h"
#include "support_functions.h"

using namespace std;
using namespace cv;


void averageImage(std::string filename, cv::Mat &finalImage, int numImages)
{
	int idx=0, divider;
	int count;
	string fullname;
	string index;
	cv::Mat img;
	cv::Mat imgAvg;
	cv::Size imageSize;

	divider = 1;
	count = 1;
	if ((int)idx / divider >= 10)
	{
		divider *= 10;
		count++;
	}
	index = string(4 - count, '0') + to_string(idx) + ".tif";
	img = imread((filename + index), IMREAD_ANYCOLOR);
	imageSize = img.size();

	imgAvg = cv::Mat(imageSize, CV_64FC3, cv::Scalar::all(0.0));

	cv::add(img, imgAvg, imgAvg, Mat(), CV_64FC3);

	for (idx = 1; idx < numImages; idx++)
	{
		//fullname = filename + ""
		if ((int)idx / divider >= 10)
		{
			divider *= 10;
			count++;
		}
		index = std::string(4 - count, '0') + to_string(idx) + ".tif";
		img = cv::imread((filename + index), IMREAD_ANYCOLOR);
		cv::add(img, imgAvg, imgAvg, Mat(), CV_64FC3);
	}
	imgAvg = imgAvg * (1.0 / numImages);

	imgAvg.convertTo(finalImage, CV_8UC3, 1.0, 0);

}	// end of averageImage

void doSomething(std::string filename, cv::Mat &img)
{
	cv::Mat tempIn, tempOut;
	tempIn = imread((filename + "Depth_Map_132-136-1.0-avg4.png"), IMREAD_ANYCOLOR);

	medianBlur(tempIn, tempIn, 3);

	std::vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	cv::imwrite((filename + "Depth_Map_132-136-1.0-avg4_med.png"), tempIn, compression_params);

	int bpstop = 0;
}



void getImageROI(cv::Mat inputImage, cv::Size ROI, cv::Mat &outputImage)
{

	unsigned int x = (inputImage.cols >> 1) - (ROI.width >> 1);
	unsigned int y = (inputImage.rows >> 1) - (ROI.height >> 1);

	cv::Rect ROI_Rect = cv::Rect(cv::Point(x, y), ROI);

	inputImage(ROI_Rect).copyTo(outputImage);



}	// end of getImageROI