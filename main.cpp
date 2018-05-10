#include "SPBPStereo.h"
#include <time.h>

int main(int argc, char* argv[]) {
	cv::Mat img1, img2;
#if 1
	img1 = cv::imread("im2.png");
	img2 = cv::imread("im6.png");
#else
	img1 = cv::imread("E:\\data\\giga_stereo\\1\\data\\0\\0000.jpg");
	img2 = cv::imread("E:\\data\\giga_stereo\\1\\data\\1\\0000.jpg");
#endif
	cv::GaussianBlur(img1, img1, cv::Size(5, 5), 2, 2);
	cv::GaussianBlur(img2, img2, cv::Size(5, 5), 2, 2);
	stereo::SPBPStereo stereo;
	cv::Rect rect(400, 200, 400, 400);
	stereo.init(stereo::StereoParam());
	//img1(rect).copyTo(img1);
	//img2(rect).copyTo(img2);
	stereo.estimate(img1, img2);
	return 0;
}