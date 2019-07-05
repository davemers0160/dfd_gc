#ifndef CALC_DFD_ERROR_H_
#define CALC_DFD_ERROR_H_


#include <cmath>
#include <cstdint>
//#include <cstdio.h>
//#include <cstdlib>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>           
#include <opencv2/highgui/highgui.hpp>     
#include <opencv2/imgproc/imgproc.hpp>  


using namespace std;



void calcError(
    cv::Mat DepthMap, 
    cv::Mat groundTruth, 
    double &NRMSE, 
    double &NMAE, 
    double &PSNR
    )
{
	int rows = DepthMap.rows;
	int cols = DepthMap.cols;
	
	cv::Mat temp_sub = cv::Mat(cv::Size(cols,rows), CV_64FC1, cv::Scalar::all(0.0));
    cv::Mat errorMat = cv::Mat(cv::Size(cols, rows), CV_64FC1, cv::Scalar::all(0.0));
	cv::Mat grayImg = cv::Mat(cv::Size(cols, rows), CV_32FC1, cv::Scalar::all(0.0));
	
    double gt_min, gt_max;

    cv::minMaxIdx(groundTruth, &gt_min, &gt_max, NULL, NULL);
    double rng = (double)std::max(gt_max - gt_min, 1.0);
    
	//cv::cvtColor(groundTruth, grayImg, CV_BGR2GRAY, 1);

	groundTruth.convertTo(grayImg, CV_32FC1, 1, 0);
	DepthMap.convertTo(DepthMap, CV_32FC1, 1, 0);

	//cv::subtract(DepthMap, grayImg, temp_sub);
	temp_sub = DepthMap - grayImg;
	cv::accumulateSquare(temp_sub, errorMat);

	//MSE = (cv::sum(errorMat)[0]) * (1.0 / (double)(rows*cols));
    double mse = (cv::mean(errorMat)[0]);
    NRMSE = std::sqrt(mse) / rng;

    NMAE = (cv::mean(cv::abs(temp_sub))[0]) / rng;
	
	PSNR = 20 * std::log10(255.0) - 10 * std::log10(NRMSE);

	//groundTruth.convertTo(groundTruth, CV_32FC1, 1, 0);
	//cv::Mat grayImg2 = grayImg.mul(grayImg,1);
	//double meanVal = cv::mean(grayImg2)[0];

	//SNR = 10 * std::log10(meanVal / MSE);

}	// end of calcError


//http://docs.opencv.org/2.4/doc/tutorials/gpu/gpu-basics-similarity/gpu-basics-similarity.html
cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	cv::Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	cv::Mat I2_2 = I2.mul(I2);        // I2^2
	cv::Mat I1_2 = I1.mul(I1);        // I1^2
	cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

	/*************************** END INITS **********************************/

	cv::Mat mu1, mu2;   // PRELIMINARY COMPUTING
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;

	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	cv::Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	cv::Mat ssim_map;
	cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	cv::Scalar mssim = cv::mean(ssim_map); // mssim = average of ssim map
	return mssim;
    
}   // end of getMSSIM



void calc_DfD_Error(
    cv::Mat DepthMap, 
    cv::Mat groundTruth,     
    double &NRMSE,
	double &PSNR,
	double &NMAE,
    double &ssim_val
    )
{
    
	//cv::Mat errorMat;

	calcError(DepthMap, groundTruth, NRMSE, NMAE, PSNR);

	cv::Scalar ssim = getMSSIM(DepthMap, groundTruth);
    ssim_val = ssim[0];

}   // end of calc_DfD_Error
    
#endif  // CALC_DFD_ERROR_H_
