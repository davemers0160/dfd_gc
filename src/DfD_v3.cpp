
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
	#include <Windows.h>
#else
	#include <pthread.h>
#endif

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <sstream>
#include <fstream>

// OpenCV Includes
#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  
//#include <opencv/cv.h>

// Cusom Includes
#include "DfD.h"
#include "support_functions.h"
#include "create_blur.h"

using namespace std;
using namespace cv;

////////////////////////////////////////////////////////////////////////////////////////////////////////////
void dfd(string image_locations, ofstream &DataLogStream, double maxSigma, double minSigma, cv::Mat infocusImage, cv::Mat defocusImage, cv::Mat &DepthMap, cv::Mat &combinedImages)
{
	int idx, jdx, kdx;
	int col, row;
	int classes = 256;

	//double section_start, section_stop;
	//double section_time;

    auto section_start = std::chrono::system_clock::now();
    auto section_stop = std::chrono::system_clock::now();
    std::chrono::duration<double, std::milli> section_time = section_stop - section_start;

	string sdate, stime;

	string logfileName; // = "DfD_v2_logfile.txt";

	vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	Size imageSize = infocusImage.size();
	row = infocusImage.rows;
	col = infocusImage.cols;

	DataLogStream << "Image Size: " << row << " x " << col << std::endl;
    std::cout << "Image Size: " << row << " x " << col << std::endl;
//	//////////////////////////////////////////////////////////////////////////////////
//	// Step 1: Read all in focus image and true defocus image  Convert the color	//
//	//		   images to YCrCb channel												//
//	//////////////////////////////////////////////////////////////////////////////////
	//getcurrenttime(sdate, stime);
	//cout << "Log File: " << logfileName << endl;
    std::cout << "Starting Step 1 ..." << std::endl;
	DataLogStream << "Starting Step 1 ..." << std::endl;

    section_start = std::chrono::system_clock::now();

	////// Convert the two images from RGB to YCrCb (Here "in" means in focus and "out" means defocus) ///
	cv::Mat YCbCrin = cv::Mat(imageSize, CV_8UC3);
    cv::Mat YCbCrout = cv::Mat(imageSize, CV_8UC3);
    cv::cvtColor(infocusImage, YCbCrin, COLOR_BGR2YCrCb, 3);
    cv::cvtColor(defocusImage, YCbCrout, COLOR_BGR2YCrCb, 3);

	std::vector<cv::Mat> YCRCB_IN(3);
	std::vector<cv::Mat> YCRCB_OUT(3);
	for (idx = 0; idx < 3; idx++)
	{
		YCRCB_IN[idx] = cv::Mat(imageSize, CV_8UC1);
		YCRCB_OUT[idx] = cv::Mat(imageSize, CV_8UC1);
	}

	//////// Split  images into 3 channels   /////////////////////////////////////////////////////////
    cv::split(YCbCrin, YCRCB_IN);
    cv::split(YCbCrout, YCRCB_OUT);

	//imwrite((image_locations + "\\yin.png"), YCRCB_IN[0]);
	//imwrite((image_locations + "\\Crin.png"), YCRCB_IN[1]);
	//imwrite((image_locations + "\\Cbin.png"), YCRCB_IN[2]);
	//imwrite((image_locations + "\\yout.png"), YCRCB_OUT[0]);
	//imwrite((image_locations + "\\Crout.png"), YCRCB_OUT[1]);
	//imwrite((image_locations + "\\Cbout.png"), YCRCB_OUT[2]);

	//section_stop = (double)cvGetTickCount();
	//section_time = (section_stop - section_start) / ((double)cvGetTickFrequency()*1000.0);
    section_stop = std::chrono::system_clock::now();
    //section_time = std::chrono::duration_cast<std::chrono::milliseconds>(section_stop - section_start);
    section_time = (section_stop - section_start);

    std::cout << "Completed Step 1 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 1 in " << section_time.count() << "ms." << std::endl;

	//////////////////////////////////////////////////////////////////////////////////
	// Step 2: Convert all pixels in Y, Cr, Cb  from integer to floating point		//
	//////////////////////////////////////////////////////////////////////////////////

    std::cout << "Starting Step 2 ..." << std::endl;
    section_start = std::chrono::system_clock::now();

    cv::Mat ImageInFocusY = cv::Mat(imageSize, CV_64FC1);
    cv::Mat ImageInFocusCr = cv::Mat(imageSize, CV_64FC1);
    cv::Mat ImageInFocusCb = cv::Mat(imageSize, CV_64FC1);
	YCRCB_IN[0].convertTo(ImageInFocusY, CV_64FC1, 1, 0);
	YCRCB_IN[1].convertTo(ImageInFocusCr, CV_64FC1, 1, 0);
	YCRCB_IN[2].convertTo(ImageInFocusCb, CV_64FC1, 1, 0);

    cv::Mat ImageOutOfFocusY = cv::Mat(imageSize, CV_64FC1);
    cv::Mat ImageOutOfFocusCr = cv::Mat(imageSize, CV_64FC1);
    cv::Mat ImageOutOfFocusCb = cv::Mat(imageSize, CV_64FC1);
	YCRCB_OUT[0].convertTo(ImageOutOfFocusY, CV_64FC1, 1, 0);
	YCRCB_OUT[1].convertTo(ImageOutOfFocusCr, CV_64FC1, 1, 0);
	YCRCB_OUT[2].convertTo(ImageOutOfFocusCb, CV_64FC1, 1, 0);

    section_stop = std::chrono::system_clock::now();
    section_time = (section_stop - section_start);

    std::cout << "Completed Step 2 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 2 in " << section_time.count() << "ms." << std::endl;

	//////////////////////////////////////////////////////////////////////////////////
	// Step 3: use highpass filter to generate edge map for each channel			//
	//////////////////////////////////////////////////////////////////////////////////

#ifdef GEN_HIGHPASS
    std::cout << "Starting Step 3 ..." << std::endl;
    section_start = chrono::system_clock::now();

    cv::Mat scr;
    cv::Mat highpassY;
    cv::Mat highpassCr;
    cv::Mat highpassCb;

    cv::Mat kernel(3,3,CV_32F,cv::Scalar(-0.8));
	kernel.at<float>(1,1) = 6.4;

	// convert the infocus image to a grayscale image
    cv::Mat infocusImage_Gray;
    cv::cvtColor(infocusImage, infocusImage_Gray, CV_BGR2GRAY, 1);

	//scr = cv::imread(image_locations + "\\view5.tif", 0);   // read in focus image(grayscale)
    cv::filter2D(infocusImage_Gray, highpassY, infocusImage_Gray.depth(), kernel);
    cv::imwrite(image_locations + "\\highpass.png", highpassY, compression_params);

	scr = cv::imread(image_locations+"\\Crin.png", 0);
    cv::filter2D(scr, highpassCr, scr.depth(), kernel);
    cv::imwrite(image_locations + "\\highpassCr.png", highpassCr, compression_params);

	scr = cv::imread(image_locations+"\\Cbin.png", 0);
    cv::filter2D(scr, highpassCb, scr.depth(), kernel);
    cv::imwrite(image_locations + "\\highpassCb.png", highpassCb, compression_params);

    section_stop = chrono::system_clock::now();
    section_time = (section_stop - section_start);

	std::cout << "Completed Step 3 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 3 in " << section_time.count() << "ms." << std::endl;

#else
	std::cout << "Skipping highpass filtering operations" << std::endl;
	DataLogStream << "Skipping highpass filtering operations" << std::endl;
#endif

	//////////////////////////////////////////////////////////////////////////////////
	// Step 4: use texture identifer to generate texture map for each channel		//
	//////////////////////////////////////////////////////////////////////////////////

#ifdef GEN_TEXTURES
	std::cout << "Starting Step 4 ..." << std::endl;
    section_start = chrono::system_clock::now();

	cv::Mat textureY;
	cv::Mat textureCr;
	cv::Mat textureCb;

	texturelessRegions(highpassY, textureY, 3, 100);
	texturelessRegions(highpassCr, textureCr, 3, 64);
	texturelessRegions(highpassCb, textureCb, 3, 64);

	cv::imwrite((image_locations + "\\texturelessregionY.png"), textureY, compression_params);
	cv::imwrite((image_locations + "\\texturelessregionCr.png"), textureCr, compression_params);
	cv::imwrite((image_locations + "\\texturelessregionCb.png"), textureCb, compression_params);

    section_stop = chrono::system_clock::now();
    section_time = (section_stop - section_start);

	std::cout << "Completed Step 4 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 4 in " << section_time.count() << "ms." << std::endl;
#else
    std::cout << "Skipping generation of textureless regions" << std::endl;
	DataLogStream << "Skipping generation of textureless regions" << std::endl;
#endif

	//////////////////////////////////////////////////////////////////////////////////
	// Step 5: use all in focus image to generate 256 synthetic defocus images		//
	//		   And use pre depth result to initial xttemp							//
	//////////////////////////////////////////////////////////////////////////////////

    std::cout << "Starting Step 5 ..." << std::endl;
    section_start = std::chrono::system_clock::now();

	////// y : true defocus image ///////////////////////////////////////////////////////////////
	////// xt : 256 synthetic defocus images ////////////////////////////////////////////////////
	//vector<Mat> xt_G(MAX_CLASSES);
	//vector<Mat> xt_Y(MAX_CLASSES);
	//vector<Mat> xt_Cr(MAX_CLASSES);
	//vector<Mat> xt_Cb(MAX_CLASSES);
	vector<cv::Mat> xt_Y;
	vector<cv::Mat> xt_Cr;
	vector<cv::Mat> xt_Cb;

	//for (idx = 0; idx < classes; idx++)
	//{
	//	//xt_G[idx] = Mat(imageSize, CV_64FC1, Scalar::all(0));
	//	xt_Y[idx] = Mat(imageSize, CV_64FC1, Scalar::all(0));
	//	xt_Cr[idx] = Mat(imageSize, CV_64FC1, Scalar::all(0));
	//	xt_Cb[idx] = Mat(imageSize, CV_64FC1, Scalar::all(0));
	//}
	//cv::Mat kernel;
	//int n = 2;
	//createGaussKernel(13, 1.2, kernel);
	//createblur(ImageInFocusY, sigma, classes, std::ref(xt_Y));

	// full YCrCB color
	double sigma_step = (maxSigma - minSigma) / (double)classes;

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
	std::thread t_Y(create_blur, ImageInFocusY, minSigma, sigma_step, classes, std::ref(xt_Y));
	std::thread t_Cr(create_blur, ImageInFocusCr, minSigma, sigma_step, classes, std::ref(xt_Cr));
	std::thread t_Cb(create_blur, ImageInFocusCb, minSigma, sigma_step, classes, std::ref(xt_Cb));
	t_Y.join();
	t_Cr.join();
	t_Cb.join();

#else
	createblur(ImageInFocusY, maxSigma, minSigma, classes, std::ref(xt_Y));
	createblur(ImageInFocusCr, maxSigma, minSigma, classes, std::ref(xt_Cr));
	createblur(ImageInFocusCb, maxSigma, minSigma, classes, std::ref(xt_Cb));
	
#endif


	// grayscale only
	//Mat infocusImage_gray, defocusImage_gray;
	//cvtColor(infocusImage, infocusImage_gray, CV_BGR2GRAY);
	//cvtColor(defocusImage, defocusImage_gray, CV_BGR2GRAY);
	//infocusImage_gray.convertTo(infocusImage_gray, CV_64FC1);
	//defocusImage_gray.convertTo(defocusImage_gray, CV_64FC1);
	//std::thread t_G(createblur, infocusImage_gray, sigma, classes, xt_G);
	//t_G.join();	
	
	classes = (int)xt_Y.size();

    section_stop = std::chrono::system_clock::now();
    section_time = (section_stop - section_start);

    std::cout << "Completed Step 5 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 5 in " << section_time.count() << "ms." << std::endl;

	//////////////////////////////////////////////////////////////////////////////////
	// Step 6: Graphcut calculation													//
	//////////////////////////////////////////////////////////////////////////////////
	//getcurrenttime(sdate, stime);
    std::cout << "Starting Step 6 ..." << std::endl;
    section_start = std::chrono::system_clock::now();

	//vector<Mat> diff_G(MAX_CLASSES);
    //vector<Mat> diff_Y(MAX_CLASSES);
    //vector<Mat> diff_Cr(MAX_CLASSES);
    //vector<Mat> diff_Cb(MAX_CLASSES);	
    std::vector<cv::Mat> diff_Y(classes);
    std::vector<cv::Mat> diff_Cr(classes);
    std::vector<cv::Mat> diff_Cb(classes);

	//// calculate Data term (y-b)^2 ////////////////////////////////////////////////////////
    std::string DataLog = "";
	int ROI_Size = 200;
    cv::Size w_size = cv::Size(ROI_Size, ROI_Size);
	int ROI_x = 0;
	int ROI_y = 0;
    cv::Rect DfD_ROI = cv::Rect(ROI_x, ROI_y, ROI_Size, ROI_Size);

	//int full_row = ((int)(imageSize.height / ROI_Size))*ROI_Size;
	//int full_col = ((int)(imageSize.width / ROI_Size))*ROI_Size;
	//Mat DepthMap = Mat(Size(full_col, full_row), CV_8UC1, Scalar::all(0));
	DepthMap = cv::Mat(imageSize, CV_8UC1, cv::Scalar::all(0));

	//Mat DepthMap_s = Mat(w_size, CV_8UC1);

	//for (ROI_y = 0; ROI_y <= (imageSize.height - ROI_Size); ROI_y += ROI_Size)
	//{
	//	for (ROI_x = 0; ROI_x <= (imageSize.width - ROI_Size); ROI_x += ROI_Size)
	//	{
	//		DfD_ROI = Rect(ROI_x, ROI_y, ROI_Size, ROI_Size);

			for (kdx = 0; kdx < classes; kdx++)
			{
				diff_Y[kdx] = cv::Mat(imageSize, CV_64FC1, cv::Scalar::all(0.0));
				diff_Cr[kdx] = cv::Mat(imageSize, CV_64FC1, cv::Scalar::all(0.0));
				diff_Cb[kdx] = cv::Mat(imageSize, CV_64FC1, cv::Scalar::all(0.0));
                cv::Mat temp_sub = cv::Mat(imageSize, CV_64FC1, cv::Scalar::all(0.0));

                cv::subtract(ImageOutOfFocusY, xt_Y[kdx], temp_sub);
                cv::accumulateSquare(temp_sub, diff_Y[kdx]);

                cv::subtract(ImageOutOfFocusCr, xt_Cr[kdx], temp_sub);
                cv::accumulateSquare(temp_sub, diff_Cr[kdx]);

                cv::subtract(ImageOutOfFocusCb, xt_Cb[kdx], temp_sub);
                cv::accumulateSquare(temp_sub, diff_Cb[kdx]);

				// grayscale version
				//diff_G[kdx] = cv::Mat(imageSize, CV_64FC1, cv::Scalar::all(0.0));				
				//cv::subtract(defocusImage_gray, xt_G[kdx], temp_sub);
				//cv::accumulateSquare(temp_sub, diff_G[kdx]);

				//diff_Y[kdx] = cv::Mat(w_size, CV_64FC1, cv::Scalar::all(0.0));
				//diff_Cr[kdx] = cv::Mat(w_size, CV_64FC1, cv::Scalar::all(0.0));
				//diff_Cb[kdx] = cv::Mat(w_size, CV_64FC1, cv::Scalar::all(0.0));
				//cv::Mat temp_sub = cv::Mat(w_size, CV_64FC1, cv::Scalar::all(0.0));

				//cv::subtract(ImageOutOfFocusY(DfD_ROI), xt_Y[kdx](DfD_ROI), temp_sub);
				//cv::accumulateSquare(temp_sub, diff_Y[kdx]);

				//cv::subtract(ImageOutOfFocusCr(DfD_ROI), xt_Cr[kdx](DfD_ROI), temp_sub);
				//cv::accumulateSquare(temp_sub, diff_Cr[kdx]);

				//cv::subtract(ImageOutOfFocusCb(DfD_ROI), xt_Cb[kdx](DfD_ROI), temp_sub);
				//cv::accumulateSquare(temp_sub, diff_Cb[kdx]);

			}	// end of kdx loop			

			//double start = (double)cvGetTickCount();

			// full YCrCb version
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

			std::thread map_t(map3, std::ref(DataLog), DepthMap, diff_Y, diff_Cr, diff_Cb, classes);		
			
			// grayscale version
			//std::thread map_t(map3, std::ref(DataLog), DepthMap, diff_G, diff_Cr, diff_Cb, classes);

			// smaller sub version
			//std::thread map_t(map3, std::ref(DataLog), DepthMap_s, diff_Y, diff_Cr, diff_Cb, classes);
			
			map_t.join();

#else
			//pthread_t map_t;
			//DFD_Thread_Vars DFD_Info;
			//pthread_create(&map_t, NULL, map3, (void*)(&videoSaveFocus) );
			map3(std::ref(DataLog), DepthMap, diff_Y, diff_Cr, diff_Cb, classes);
			
#endif	


	//		DepthMap_s.copyTo(DepthMap(DfD_ROI));

	//	}

	//}

    std::cout << "MAP Complete." << std::endl;
	DataLogStream << DataLog;

	//cout << "Saving Depth Map..." << endl;
	//combinedImages = Mat(DepthMap.rows, DepthMap.cols * 2, CV_8UC1);
	//YCRCB_IN[0].copyTo(combinedImages(Rect(0, 0, DepthMap.cols, DepthMap.rows)));
	//DepthMap.copyTo(combinedImages(Rect(DepthMap.cols, 0, DepthMap.cols, DepthMap.rows)));
	////imwrite(image_locations + "\\Depth_Map.png", DepthMap, compression_params);
	//imwrite(image_locations + "\\Combined_Images.png", combinedImages, compression_params);

    section_stop = std::chrono::system_clock::now();
    section_time = (section_stop - section_start);

    std::cout << "Completed Step 6 in " << section_time.count() << "ms." << std::endl;
	DataLogStream << "Completed Step 6 in " << section_time.count() << "ms." << endl << std::endl;
	//getcurrenttime(sdate, stime);

	//DataLogStream.close();
	//Beep(500, 1000);
}


