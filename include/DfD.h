#ifndef DFD_H
#define DFD_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  

//#include "create_blur.h"

/////////////////////////////////// DEFINES ///////////////////////////////////
//#define MAX_CLASSES		256		/* Max number of classes allowed */
//#define MAX_SIGMA		2.5		/* max sigma used for the Gaussian blur settings */
#define PI				3.14159265358979323846
// used to generate the synthetic blur from a given depth map
//#define GEN_DEFOCUS_IMAGE

// used to generate the high pass filtered image
//#define GEN_HIGHPASS

// used to generate the textureless regions
//#define GEN_TEXTURES
#ifdef GEN_TEXTURES
	#define GEN_HIGHPASS
#endif

// use this to define the use of the bias image to try and remove noise
//#define USE_BIAS

// use this to apply the dark and flat corrections.
//#define APPLY_CORRECTIONS


using namespace std;
using namespace cv;

////////////////////////////////// Typedefs //////////////////////////////////
typedef struct
{
	string DataLog;
	Mat Depth_Map;
	vector<Mat> diff_Y;
	vector<Mat> diff_Cr;
	vector<Mat> diff_Cb;
	int classes;
	
} DFD_Thread_Vars;



////////////////////////////////// FUNCTIONS //////////////////////////////////
void dfd(string image_locations, ofstream &DataLogStream, double maxSigma, double minSigma, cv::Mat infocusImage, cv::Mat defocusImage, cv::Mat &DepthMap, cv::Mat &combinedImages);
void readFile(Mat &img1, Mat &img2, string f1, string f2, int type, int num);
void parseCSVFile(std::string parseFilename, std::vector<std::string> &vFocusFile, std::vector<std::string> &vDefocusFile, std::vector<std::string> &vSaveLocation);

//if defined(_WIN32) | defined(__WIN32__) | defined(__WIN32) | defined(_WIN64) | defined(__WIN64)

void map3(string &DataLog, Mat &Depth_Map, vector<cv::Mat> &diff_Y, vector<cv::Mat> &diff_Cr, vector<cv::Mat> &diff_Cb, int classes);

//#else
	
//void *map3(void *args);

//#endif
//void createblur(Mat ImageInFocus, double sigma, int classes, vector<Mat> &xt);
//int createGaussKernel(int size, double sigma, cv::Mat &kernel);
void diffSum(int classes, std::vector<cv::Mat> &diff_Y, std::vector<cv::Mat> &diff_Cr, std::vector<cv::Mat> &diff_Cb, std::vector<cv::Mat> &logpost);
void texturelessRegions(cv::Mat &inputImage, cv::Mat &textureImage, int windowSize, int thresh);
void gen_smooth_terms(int32_t classes, int32_t smooth[]);
void GridGraph_DArraySArray(int width, int height, int num_labels, std::vector<cv::Mat> &diffSumMat, cv::Mat &gridResult, std::string &DataLog);

void calcError(cv::Mat DepthMap, cv::Mat groundTruth, cv::Mat &errorMat, double &MSE, double &SNR, double &PSNR);
Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2);

#endif	// end of DFD_H 