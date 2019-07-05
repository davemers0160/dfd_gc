#ifndef SUPPORT_FCN_H
#define SUPPORT_FCN_H

#include <vector>
#include <cstdlib>
#include <string>

#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  

/////////////////////////////////// DEFINES ///////////////////////////////////


using namespace std;
using namespace cv;

////////////////////////////////// FUNCTIONS //////////////////////////////////
void averageImage(string filename, Mat &finalImage, int numImages);
void doSomething(string filename, Mat &img);
void getImageROI(cv::Mat inputImage, cv::Size ROI, cv::Mat &outputImage);

#endif	// end of SUPPORT_FCN_H 
