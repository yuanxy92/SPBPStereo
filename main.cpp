#include "SPBPStereo.h"
#include <time.h>

int main(int argc, char* argv[]) {
	srand(12345);
	//cv::Mat img1 = cv::imread("im2.png");
	cv::Mat img1 = cv::imread("E:\\data\\giga_stereo\\1\\data\\0\\0000.jpg");
	//cv::Mat img2 = cv::imread("im6.png");
	cv::Mat img2 = cv::imread("E:\\data\\giga_stereo\\1\\data\\1\\0000.jpg");
	stereo::SPBPStereo stereo;
	cv::Rect rect(400, 200, 400, 400);
	stereo.init(stereo::StereoParam());
	//img1(rect).copyTo(img1);
	//img2(rect).copyTo(img2);
	stereo.estimate(img1, img2);
	return 0;
}