#include "SPBPStereo.h"

int main(int argc, char* argv[]) {
	cv::Mat img1 = cv::imread("E:\\data\\giga_stereo\\1\\data\\0\\0000.jpg");
	cv::Mat img2 = cv::imread("E:\\data\\giga_stereo\\1\\data\\1\\0000.jpg");
	stereo::SPBPStereo stereo;
	stereo.init(stereo::StereoParam());
	stereo.estimate(img1, img2);
	return 0;
}