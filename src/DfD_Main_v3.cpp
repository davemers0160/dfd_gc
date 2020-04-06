#define _CRT_SECURE_NO_WARNINGS

#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
	#include <Windows.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <thread>
#include <sstream>
#include <fstream>
#include <chrono>
#include <string>

// OpenCV includes
#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/video/video.hpp>

// Custom includes
//#include "create_blur.h"
#include "get_platform.h"
#include "getopt.h"
#include "DfD.h"
#include "support_functions.h"
#include "get_current_time.h"
#include "file_ops.h"
#include "file_parser.h"
#include "calc_dfd_error.h"
//#include "path_check.h"


//#define USE_VIDEO

#define SINGLE_PAIR		0	/*Process a single pari of images*/
#define AVI_PARSE		1
#define TIME_AVG		2	/*Time average of image files*/
#define TIME_MEDIAN		3	/*Time medain of images*/
#define DARK_HIST		4	/*Process the histogram of the dark image*/

using namespace std;
using namespace cv;

//-------------------------------GLOBALS---------------------------------------
std::string platform;

void print_usage(void)
{
	std::cout << "Enter the following as arguments into the program:" << std::endl;
	std::cout << "<input file w/ file paths for infocus, defocus & groundtruth> <output directory> <sigma range>" << std::endl;
    std::cout << "-f ../test_input_miya_room.txt -o ../results/miya_room_ell2/ -s 1.25:0.25:2.50" << std::endl;
	std::cout << endl;

}

//-----------------------------------------------------------------------------
///////////////////////////////////////////////////////////////////////////////
//-----------------------------------------------------------------------------

int main(int argc, char** argv)
{
	int32_t idx, jdx, kdx;
	int32_t classes = 256;
	int32_t avg = 1;
	int8_t c;
    int32_t n=0;      // index to read a single file

    std::vector<double> sigma = { 2.56 }; // = { 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5 };
    std::vector<double> minSigma = { 0.32 };

	cv::Mat infocusImage, defocusImage;
	cv::Mat groundTruth;
	cv::Mat combinedImages;
    cv::Mat DepthMap;
        
    std::string data_directory;
    std::string focusfilename, defocusfilename;
	std::string SaveLocation;
	std::string groundTruthfilename;
	std::string sdate, stime;
	std::string loc_substr;

	std::ofstream DataLogStream;
	std::string logfileName;// = "DfD_v2_logfile.txt";
	std::string image_locations;
	//std::string biasFilename;
	//std::string darkFilename;
	//std::string flatFilename;
	std::string parseFilename;


	// read in still images and average N images
	//cv::Mat tempF, tempD;
	//cv::Size ROI_Size;

	std::vector<int> compression_params;
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(0);

	// variables to read in data from csv file
	std::vector<std::string> vFocusFile;
	std::vector<std::string> vDefocusFile;
    std::vector<std::string> vDepthFile;
    std::vector<std::string> vSaveLocation;

    std::vector<std::vector<std::string>> params;

	bool parsefile = false;
    bool single_file = false;

    typedef std::chrono::duration<double> d_sec;
    auto start_time = chrono::system_clock::now();
    auto stop_time = chrono::system_clock::now();
    auto elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

	//////////////////////////////////////////////////////////////////////////////////

	if (argc == 1)
	{
		print_usage();
		std::cin.ignore();
		return 0;
	}

    // sample input: -f ../test_input_miya_room.txt -o ../results/miya_room_ell2/ -s 1.0:0.25:2.0
	while ((c = getopt(argc, argv, "hl:i:d:o:f:a:s:n:m:")) != -1)
	{
		switch (c)
		{
		case 'h':				// help
			print_usage();
			return 0;
			break;

		//case 'l':				// input image directory 
		//	image_locations = path_check(optarg);
		//	break;

		//case 'i':				// infocus image
		//	focusfilename = optarg;
		//	break;

		//case 'd':				// defocus image 
		//	defocusfilename = optarg;
		//	break;
		case 'f':				// read in image pairs from a csv file
			parseFilename = optarg;
			parsefile = true;
			break;
		//case 'a':
		//	avg = atoi(optarg);
		//	break;
		case 'o':				// output file directory
			SaveLocation = path_check(optarg);
            break;

        case 's':               // sigma range [min_sigma:step:max_sigma]       
            parse_input_range(optarg, sigma);
            break;
         
        case 'n':               // use this option to supply multiple items but only run one
            n = atoi(optarg);
            single_file = true;
            break;

        //case 'm':
        //    parse_input_range(optarg, minSigma);
        //    break;

		default:
			break;
		}
	}

	if (parsefile == true)
	{
        parse_csv_file(parseFilename, params);// vFocusFile, vDefocusFile, vSaveLocation);
        
        data_directory = path_check(params[0][0]);
        params.erase(params.begin());
        
        std::cout << "Input Image Count: " << params.size() << std::endl;

        if(single_file==false)
        {
            for (idx = 0; idx < params.size(); ++idx)
            {
                vFocusFile.push_back(data_directory + params[idx][0]);
                vDefocusFile.push_back(data_directory + params[idx][1]);
                vDepthFile.push_back(data_directory + params[idx][2]);
            }
        }
        else
        {
            vFocusFile.push_back(data_directory + params[n][0]);
            vDefocusFile.push_back(data_directory + params[n][1]);
            vDepthFile.push_back(data_directory + params[n][2]);
        }

	}
	else
	{
		// read in multiple files to time average
		readFile(infocusImage, defocusImage, (image_locations + focusfilename), (image_locations + defocusfilename), TIME_MEDIAN, avg);
	}

    // run a check of the sigma values
    if (sigma.size() != minSigma.size())
    {
        std::cout << "Warning: The number of minimum sigma values does not equal the number of sigma values!" << std::endl;
        if (sigma.size() > minSigma.size())
        {
            std::cout << "          Expanding The number of minimum sigma values to match the number of sigma values." << std::endl;
            uint64_t s = minSigma.size() - 1;
            for (uint64_t mdx = minSigma.size(); mdx < sigma.size(); ++mdx)
            {
                minSigma.push_back(minSigma[s]);
            }
        }
        else if (sigma.size() < minSigma.size())
        {
            std::cout << "          Only the first "<< sigma.size() << " minimum sigma values will be used." << std::endl;
        }       
    }


    get_platform(platform);

	get_current_time(sdate, stime);
    char intStr[5];
    sprintf(intStr, "%03d", n);
    string n_str = string(intStr);
	logfileName = "DfD_v2_logfile_" +  n_str + "_" + sdate + "_" + stime + ".txt";
    DataLogStream.open((SaveLocation + logfileName), ios::out);

    std::cout << "Platform:          " << platform << std::endl;
    DataLogStream << "Platform:          " << platform << std::endl;

	//////////////////////////////////////////////////////////////////////////////////
	// Step 1: Read all in focus image and true defocus image  Convert the color	//
	//		   images to YCrCb channel												//
	//////////////////////////////////////////////////////////////////////////////////

	for (idx = 0; idx < vFocusFile.size(); idx++)
	{
		// get the save location
        std::string tmpSaveLocation = SaveLocation;
        if (tmpSaveLocation == "")
        {
            tmpSaveLocation = vSaveLocation[idx];
        }

        tmpSaveLocation = path_check(tmpSaveLocation);

        int focus_loc, defocus_loc, groundtruth_loc;

		// get the location of the start of the filename
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
        focus_loc = ((int)vFocusFile[idx].rfind('\\') == -1) ? ((int)vFocusFile[idx].rfind('/')) : ((int)vFocusFile[idx].rfind('\\'));
        defocus_loc = ((int)vDefocusFile[idx].rfind('\\') == -1) ? ((int)vDefocusFile[idx].rfind('/')) : ((int)vDefocusFile[idx].rfind('\\'));
        groundtruth_loc = ((int)vDepthFile[idx].rfind('\\') == -1) ? ((int)vDepthFile[idx].rfind('/')) : ((int)vDepthFile[idx].rfind('\\'));
#else
        focus_loc = (int)vFocusFile[idx].rfind('/');
        defocus_loc = (int)vDefocusFile[idx].rfind('/');
        groundtruth_loc = (int)vDepthFile[idx].rfind('/');
#endif

		// extract just the filename for saving later
		focusfilename = vFocusFile[idx].substr(focus_loc + 1, vFocusFile[idx].length() - 1);
		defocusfilename = vDefocusFile[idx].substr(defocus_loc + 1, vDefocusFile[idx].length() - 1);
        groundTruthfilename = vDepthFile[idx].substr(groundtruth_loc + 1, vDepthFile[idx].length() - 1);

		readFile(infocusImage, defocusImage, vFocusFile[idx], vDefocusFile[idx], SINGLE_PAIR, avg);

        cv::Mat groundTruth = imread(vDepthFile[idx], IMREAD_GRAYSCALE);

        // check to see if the images have been read correctly and that there is data in the images
        if (infocusImage.empty() == true) 
        {
            std::cout << std::endl << "------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Error: " << focusfilename << " is empty!" << std::endl;
            DataLogStream << "Error: " << focusfilename << " is empty!" << std::endl;
            std::cout << "------------------------------------------------------------------------------------" << std::endl;
            break;
        }
        if (defocusImage.empty() == true)
        {
            std::cout << std::endl << "------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Error: " << defocusfilename << " is empty!" << std::endl;
            DataLogStream << "Error: " << defocusfilename << " is empty!" << std::endl;
            std::cout << "------------------------------------------------------------------------------------" << std::endl;
            break;
        }
        
        if (groundTruth.empty() == true)
        {
            std::cout << std::endl << "------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Error: " << groundTruthfilename << " is empty!" << std::endl;
            DataLogStream << "Error: " << groundTruthfilename << " is empty!" << std::endl;
            std::cout << "------------------------------------------------------------------------------------" << std::endl;
            break;
        }

        // check to make sure that the images are all the same size....
        if ((infocusImage.size() != defocusImage.size()) || (infocusImage.size() != groundTruth.size()) || (defocusImage.size() != groundTruth.size()))
        {
            std::cout << std::endl << "------------------------------------------------------------------------------------" << std::endl;
            std::cout << "Error: Image sizes are not all the same! " << std::endl;
            std::cout << focusfilename << ": " << infocusImage.size().width << " x " << infocusImage.size().height << std::endl;
            std::cout << defocusfilename << ": " << defocusImage.size().width << " x " << defocusImage.size().height << std::endl;
            std::cout << groundTruthfilename << ": " << groundTruth.size().width << " x " << groundTruth.size().height << std::endl;
            std::cout << "------------------------------------------------------------------------------------" << std::endl;
            
            DataLogStream << focusfilename << ": " << infocusImage.size().width << " x " << infocusImage.size().height << std::endl;
            DataLogStream << defocusfilename << ": " << defocusImage.size().width << " x " << defocusImage.size().height << std::endl;
            DataLogStream << groundTruthfilename << ": " << groundTruth.size().width << " x " << groundTruth.size().height << std::endl;

            break;
        }

		// Create a crop of the input image for whatever reason
        // cv::Size ROI_Size = cv::Size(infocusImage.cols-20, infocusImage.rows - 2);
		// cv::Size ROI_Size = cv::Size(768, 512);
		// getImageROI(infocusImage, ROI_Size, infocusImage);
		// getImageROI(defocusImage, ROI_Size, defocusImage);
        // getImageROI(groundTruth, ROI_Size, groundTruth);

//------------------------------------------------------------------------------------
        // Do a blur of the input images except the ground truth to test a theory
        //int row, col;
        //int size = 19;
        //double s = 0.5*0.5;
        //cv::Mat kernel = cv::Mat::zeros(size, size, CV_32FC1);

        //for (row = 0; row < size; ++row)
        //{
        //    for (col = 0; col < size; ++col)
        //    {
        //        kernel.at<float>(row, col) = (1.0 / (2 * CV_PI *s)) * std::exp((-((col - (size >> 1))*(col - (size >> 1))) - ((row - (size >> 1))*(row - (size >> 1)))) / (2 * s));
        //    }
        //}

        //double matsum = (double)cv::sum(kernel)[0];

        //kernel = kernel * (1.0 / matsum);	// get the matrix to sum up to 1...
        //cv::filter2D(infocusImage, infocusImage, CV_32FC1, kernel, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_REPLICATE);
        //cv::filter2D(defocusImage, defocusImage, CV_32FC1, kernel, cv::Point(-1, -1), 0.0, cv::BorderTypes::BORDER_REPLICATE);

//------------------------------------------------------------------------------------


        std::cout << "Infocus Image:     " << vFocusFile[idx] << std::endl;
        std::cout << "Defocus Image:     " << vDefocusFile[idx] << std::endl;
        std::cout << "Groundtruth Image: " << vDepthFile[idx] << std::endl;

        DataLogStream << "Infocus Image:     " << vFocusFile[idx] << std::endl;
        DataLogStream << "Defocus Image:     " << vDefocusFile[idx] << std::endl;
        DataLogStream << "Groundtruth Image: " << vDepthFile[idx] << std::endl << std::endl;

		std::cout << "Logfile Name:      " << logfileName << std::endl << std::endl;

		std::size_t infocus_ext_loc = focusfilename.rfind('.');
		std::size_t defocus_ext_loc = defocusfilename.rfind('.');

        std::string defocus_add = defocusfilename.substr(0, defocus_ext_loc);

        std::string blur_type = "_lin2_";
        //std::string blur_type = "_ps1_";

        double NRMSE, PSNR, NMAE, ssim_val;

        for (int jdx = 0; jdx < sigma.size(); ++jdx)
        {

            std::ostringstream sig_str;
            sig_str << std::fixed << std::setprecision(2) << std::setfill('0') << std::setw(2) << sigma[jdx];

            get_current_time(sdate, stime);
            std::cout << "Start Time: " << stime << std::endl;
            DataLogStream << "Start Time: " << stime << std::endl;

            std::cout << "Min Sigma: " << std::fixed << std::setprecision(2) << minSigma[jdx] << std::endl;
            DataLogStream << "Min Sigma: " << std::fixed << std::setprecision(2) << minSigma[jdx] << std::endl;

            std::cout << "Max Sigma: " << std::fixed << std::setprecision(2) << (sigma[jdx] + minSigma[jdx]) << std::endl;
            DataLogStream << "Max Sigma: " << std::fixed << std::setprecision(2) << (sigma[jdx] + minSigma[jdx]) << std::endl;

            start_time = chrono::system_clock::now();


            dfd(tmpSaveLocation, std::ref(DataLogStream), (sigma[jdx] + minSigma[jdx]), minSigma[jdx], infocusImage, defocusImage, DepthMap, combinedImages);
            stop_time = chrono::system_clock::now();
            elapsed_time = chrono::duration_cast<d_sec>(stop_time - start_time);

            cout << "Depth Map Generation Complete (minutes): " << elapsed_time.count()/60 << endl << endl;
            DataLogStream << "Depth Map Generation Complete (minutes): " << elapsed_time.count() / 60 << endl << endl;

            // run the error checking on the results
            calc_DfD_Error(DepthMap, groundTruth, NRMSE, PSNR, NMAE, ssim_val);

            std::cout << "------------------------------------------------------------------" << std::endl;
            DataLogStream << "------------------------------------------------------------------" << std::endl;
            
            //std::cout << "MSE: " << std::fixed << std::setprecision(6) << NRMSE << "\tMAE: " << NMAE << "\tPSNR: " << PSNR << "\tSSIM: " << ssim_val << std::endl;
            std::cout << "<NMAE>, <NRMSE>, <SSIM>: " << std::fixed << std::setprecision(6) << NMAE << ", " << NRMSE << ", " << ssim_val << std::endl;

            //DataLogStream << "MSE: " << std::fixed << std::setprecision(6) << NRMSE << "\tMAE: " << NMAE << "\tPSNR: " << PSNR << "\tSSIM: " << ssim_val << std::endl;
            DataLogStream << "<NMAE>, <NRMSE>, <SSIM>: " << std::fixed << std::setprecision(6) << NMAE << ", " << NRMSE  << ", " << ssim_val << std::endl;

            cv::cvtColor(DepthMap, DepthMap, cv::COLOR_GRAY2BGR);

            combinedImages = cv::Mat(DepthMap.rows, DepthMap.cols * 2, CV_8UC3);
            infocusImage.copyTo(combinedImages(cv::Rect(0, 0, DepthMap.cols, DepthMap.rows)));
            DepthMap.copyTo(combinedImages(cv::Rect(DepthMap.cols, 0, DepthMap.cols, DepthMap.rows)));

            cv::imwrite(tmpSaveLocation + "Depth_Map_" + blur_type + sig_str.str() + "_" + defocusfilename.substr(0, defocus_ext_loc) + ".png", DepthMap, compression_params);
            cv::imwrite(tmpSaveLocation + "Combined_Image_" + blur_type + sig_str.str() + "_" + defocusfilename.substr(0, defocus_ext_loc) + ".png", combinedImages, compression_params);

            std::cout << "------------------------------------------------------------------" << std::endl << std::endl;
            DataLogStream << "------------------------------------------------------------------" << std::endl << std::endl;

        }   // sigma range loop

	}
	
#if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)
	Beep(500, 1000);
#endif

	//string originalWindow = "Original Image";
	//namedWindow(originalWindow, WINDOW_NORMAL | WINDOW_KEEPRATIO);
	//imshow(originalWindow, infocusImage);

    if (platform != "HPC")
    {
        if (!combinedImages.empty())
        {
            std::string depthmapWindow = "Original Image/Depth Map";
            cv::namedWindow(depthmapWindow, WINDOW_NORMAL | WINDOW_KEEPRATIO);
            cv::imshow(depthmapWindow, combinedImages);
            cv::waitKey(-1);
        }
        cv::destroyAllWindows();
    }

	std::cout << "End of Program.  Press Enter to close..." << std::endl;
	DataLogStream.close();


	std::cin.ignore();
    return 0; 

}


