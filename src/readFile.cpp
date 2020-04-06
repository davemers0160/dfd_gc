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

#include "DfD.h"
#include "support_functions.h"

using namespace std;
using namespace cv;



void readFile(Mat &infocusImage, Mat &defocusImage, string f1, string f2, int type, int num)
{
	int idx, jdx, kdx;
	int bp_stop = 0;
	int lower, upper;
	Mat img1, img2;
	Size imageSize; 	
	string file1, file2;

	Mat imgAvg_I, imgAvg_O;

	Mat imgDiff;
	Mat noise; 

	// histogram variables
	int histSize = 256;
	Mat histB, histG, histR;
	float range[] = { 0.0, (float)histSize };
	const float *ranges[] = { range };
	int channels[] = { 0, 1, 2 };
	vector<Mat> BGR_Vec(3);
	ofstream histB_file, histG_file, histR_file;
	ofstream darkR, darkG, darkB;

	// median variables
	vector< vector<int> > medianVector_I(3, vector<int>(num));
	vector< vector<int> > medianVector_O(3, vector<int>(num));
	vector<Mat> medMat_I(num);
	vector<Mat> medMat_O(num);

	std::size_t f1_ext_loc, f2_ext_loc;
	string f1_ext;

	switch (type)
	{
		case 0:		// two individual files

			// img1 = cv::imread(f1, CV_LOAD_IMAGE_COLOR);
			// img2 = cv::imread(f2, CV_LOAD_IMAGE_COLOR);

			// noise = cv:::Mat(img1.size(), CV_16SC3, cv::Scalar::all(0));
			// lower = -1;
			// upper = 2;
			
			// randu(noise, lower, upper);
			// img1.convertTo(img1, CV_16SC3);
			// img1 = img1 + noise;

			// cv::randu(noise, lower, upper);
			// img2.convertTo(img2, CV_16SC3);
			// img2 = img2 + noise;

			// cv::normalize(img1, img1, 0, 255, CV_MINMAX, CV_8UC3);
			// cv::normalize(img2, img2, 0, 255, CV_MINMAX, CV_8UC3);

			// img1 = cv::max(img1, 0);
			// img1 = cv::min(img1, 255);

			// img2 = cv::max(img2, 0);
			// img2 = cv::min(img2, 255);

			// img1.convertTo(infocusImage, CV_8UC3, 1, 0);
			// img2.convertTo(defocusImage, CV_8UC3, 1, 0);
            
			infocusImage = cv::imread(f1, IMREAD_COLOR);
			defocusImage = cv::imread(f2, IMREAD_COLOR);

			break;

		case 1:		// avi 


			break;


		case 2:		// time average files

			averageImage(f1, infocusImage, num);
			averageImage(f2, defocusImage, num);

			break;

		case 3:			// time median of files

			if (num > 10)
			{
				num = 10;
			}

			// temp read of the file to get the image information
			//file1 = f1 + "0000.png";
			//file1 = f1 + "0000.tif";
			//img1 = imread(file1, CV_LOAD_IMAGE_COLOR);
			img1 = cv::imread(f1, IMREAD_COLOR);
			imageSize = img1.size();

			infocusImage = cv::Mat(imageSize, CV_8UC3, cv::Scalar::all(0));
			defocusImage = cv::Mat(imageSize, CV_8UC3, cv::Scalar::all(0));
			imgAvg_I = cv::Mat(imageSize, CV_8UC3, cv::Scalar::all(0.0));
			imgAvg_O = cv::Mat(imageSize, CV_8UC3, cv::Scalar::all(0.0));

			// parse the file name to get the extenstion and then replace the numbering to load multiple files
			f1_ext_loc = f1.rfind('.');
			f2_ext_loc = f2.rfind('.');

			//f1_ext = f1.substr(f1_ext_loc+1, f1.length() - 1);
			
			for (idx = 0; idx < num; idx++)
			{
				// assume that the last character before the '.' is a number that increments in the file name			
				file1 = f1.replace(f1_ext_loc - 1, 1, to_string(idx));
				file2 = f2.replace(f2_ext_loc - 1, 1, to_string(idx));
				//file1 = f1 + "000" + to_string(idx) + ".png";
				//file2 = f2 + "000" + to_string(idx) + ".png";
				//file1 = f1 + "000" + to_string(idx) + ".tif";
				//file2 = f2 + "000" + to_string(idx) + ".tif";
				medMat_I[idx] = cv::imread(file1, IMREAD_COLOR);
				medMat_O[idx] = cv::imread(file2, IMREAD_COLOR);

			}

			for (idx = 0; idx < medMat_I[0].rows; idx++)
			{
				for (jdx = 0; jdx < medMat_I[0].cols; jdx++)
				{

					for (kdx = 0; kdx < num; kdx++)
					{
						//bgrPixel.val[0] = pixelPtr[i*foo.cols*cn + j*cn + 0]; // B
						//bgrPixel.val[1] = pixelPtr[i*foo.cols*cn + j*cn + 1]; // G
						//bgrPixel.val[2] = pixelPtr[i*foo.cols*cn + j*cn + 2]; // R
						medianVector_I[0][kdx] = medMat_I[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 0];
						medianVector_O[0][kdx] = medMat_O[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 0];
						medianVector_I[1][kdx] = medMat_I[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 1];
						medianVector_O[1][kdx] = medMat_O[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 1];
						medianVector_I[2][kdx] = medMat_I[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 2];
						medianVector_O[2][kdx] = medMat_O[kdx].data[idx*medMat_I[0].cols * 3 + jdx * 3 + 2];
					}

					// sort full vector
					//std::sort(medianVector_I[0].begin(), medianVector_I[0].end());
					//std::sort(medianVector_O[0].begin(), medianVector_O[0].end());
					//std::sort(medianVector_I[1].begin(), medianVector_I[1].end());
					//std::sort(medianVector_O[1].begin(), medianVector_O[1].end());
					//std::sort(medianVector_I[2].begin(), medianVector_I[2].end());
					//std::sort(medianVector_O[2].begin(), medianVector_O[2].end());

					// sort up the the middle point
					std::nth_element(medianVector_I[0].begin(), medianVector_I[0].begin() + (int)(medianVector_I[0].size() / 2 + 0.5), medianVector_I[0].end());
					std::nth_element(medianVector_O[0].begin(), medianVector_O[0].begin() + (int)(medianVector_O[0].size() / 2 + 0.5), medianVector_O[0].end());
					std::nth_element(medianVector_I[1].begin(), medianVector_I[1].begin() + (int)(medianVector_I[1].size() / 2 + 0.5), medianVector_I[1].end());
					std::nth_element(medianVector_O[1].begin(), medianVector_O[1].begin() + (int)(medianVector_O[1].size() / 2 + 0.5), medianVector_O[1].end());
					std::nth_element(medianVector_I[2].begin(), medianVector_I[2].begin() + (int)(medianVector_I[2].size() / 2 + 0.5), medianVector_I[2].end());
					std::nth_element(medianVector_O[2].begin(), medianVector_O[2].begin() + (int)(medianVector_O[2].size() / 2 + 0.5), medianVector_O[2].end());


					//infocusImage.at<unsigned char>(idx, jdx) = medianVector_I[(int)num / 2][0];
					//defocusImage.at<unsigned char>(idx, jdx) = medianVector_O[(int)num / 2][0];
					infocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 0] = medianVector_I[0][(int)(num / 2 + 0.5)];
					infocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 1] = medianVector_I[1][(int)(num / 2 + 0.5)];
					infocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 2] = medianVector_I[2][(int)(num / 2 + 0.5)];
					defocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 0] = medianVector_O[0][(int)(num / 2 + 0.5)];
					defocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 1] = medianVector_O[1][(int)(num / 2 + 0.5)];
					defocusImage.data[idx*medMat_I[0].cols * 3 + jdx * 3 + 2] = medianVector_O[2][(int)(num / 2 + 0.5)];
				}
			}

			bp_stop = 1;
			break;

		case 4:			// run histogram on dark image

			img1 = imread(f1, IMREAD_COLOR);

			split(img1, BGR_Vec);

			calcHist(&BGR_Vec[0], 1, 0, Mat(), histB, 1, &histSize, ranges, true, false);
			calcHist(&BGR_Vec[1], 1, 0, Mat(), histG, 1, &histSize, ranges, true, false);
			calcHist(&BGR_Vec[2], 1, 0, Mat(), histR, 1, &histSize, ranges, true, false);

			// need to save the hist mat to a file --  not an image
			histB_file.open("histB.csv", ios_base::out);
			histG_file.open("histG.csv", ios_base::out);
			histR_file.open("histR.csv", ios_base::out);

			for (idx = 0; idx < histSize; idx++)
			{
				histB_file << idx << ", " << histB.at<float>(idx, 0) << endl;
				histG_file << idx << ", " << histG.at<float>(idx, 0) << endl;
				histR_file << idx << ", " << histR.at<float>(idx, 0) << endl;

			}
			histB_file.close();
			histG_file.close();
			histR_file.close();

			darkR.open("darkR.csv", ios_base::out);
			darkG.open("darkG.csv", ios_base::out);
			darkB.open("darkB.csv", ios_base::out);
			unsigned char temp;
			for (idx = 0; idx < BGR_Vec[0].rows; idx++)
			{
				for (jdx = 0; jdx < BGR_Vec[0].cols; jdx++)
				{
					temp = BGR_Vec[0].at<unsigned char>(idx, jdx);
					darkB << (int)BGR_Vec[0].at<unsigned char>(idx, jdx) << ", ";
					darkG << (int)BGR_Vec[1].at<unsigned char>(idx, jdx) << ", ";
					darkR << (int)BGR_Vec[2].at<unsigned char>(idx, jdx) << ", ";

				}

				darkB << endl;
				darkG << endl;
				darkR << endl;

			}

			darkB.close();
			darkG.close();
			darkR.close();




			bp_stop = 1;
			break;

		default:


			break;

	}	// end of switch

}	// end of readFile

/*
void parseCSVFile(std::string parseFilename, std::vector<std::string> &vFocusFile, std::vector<std::string> &vDefocusFile, std::vector<std::string> &vSaveLocation)
{
	std::ifstream csvfile(parseFilename);
	std::string nextLine;

	while (std::getline(csvfile, nextLine))
	{
		stringstream ss(nextLine);
		while (ss.good())
		{
			string substr;
			getline(ss, substr, ',');
			vFocusFile.push_back(substr);
			getline(ss, substr, ',');
			vDefocusFile.push_back(substr);		
			getline(ss, substr, ',');
			vSaveLocation.push_back(substr);
		}

	}


}	// end of parseCSVFile

*/
