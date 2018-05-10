/**
@brief stereo class for visualization
@author Shane Yuan
@date May 08, 2018
*/

#include "Visualizer.h"

stereo::Visualizer::Visualizer() : ncols(0) {}
stereo::Visualizer::~Visualizer() {}

void stereo::Visualizer::setcols(int r, int g, int b, int k) {
	colorwheel[k][0] = r;
	colorwheel[k][1] = g;
	colorwheel[k][2] = b;
}
void stereo::Visualizer::makecolorwheel() {
	// relative lengths of color transitions:
	// these are chosen based on perceptual similarity
	// (e.g. one can distinguish more shades between red and yellow 
	//  than between yellow and green)
	int RY = 15;
	int YG = 6;
	int GC = 4;
	int CB = 11;
	int BM = 13;
	int MR = 6;
	ncols = RY + YG + GC + CB + BM + MR;
	//printf("ncols = %d\n", ncols);
	if (ncols > MAXCOLS)
		exit(1);
	int i;
	int k = 0;
	for (i = 0; i < RY; i++) setcols(255, 255 * i / RY, 0, k++);
	for (i = 0; i < YG; i++) setcols(255 - 255 * i / YG, 255, 0, k++);
	for (i = 0; i < GC; i++) setcols(0, 255, 255 * i / GC, k++);
	for (i = 0; i < CB; i++) setcols(0, 255 - 255 * i / CB, 255, k++);
	for (i = 0; i < BM; i++) setcols(255 * i / BM, 0, 255, k++);
	for (i = 0; i < MR; i++) setcols(255, 0, 255 - 255 * i / MR, k++);
}

void stereo::Visualizer::computeColor(float fx, float fy, uchar *pix) {
	if (ncols == 0)
		makecolorwheel();
	float rad = sqrt(fx * fx + fy * fy);
	float a = atan2(-fy, -fx) / M_PI;
	float fk = (a + 1.0) / 2.0 * (ncols - 1);
	int k0 = (int)fk;
	int k1 = (k0 + 1) % ncols;
	float f = fk - k0;
	for (int b = 0; b < 3; b++) {
		float col0 = colorwheel[k0][b] / 255.0;
		float col1 = colorwheel[k1][b] / 255.0;
		float col = (1 - f) * col0 + f * col1;
		if (rad <= 1)
			col = 1 - rad * (1 - col); // increase saturation with radius
		else
			col *= .75; // out of range
		pix[2 - b] = (int)(255.0 * col);
	}
}

cv::Mat stereo::Visualizer::disparityToFlow(cv::Mat disparityMap) {
	cv::Mat flow(disparityMap.rows, disparityMap.cols, CV_32FC2);
	for (size_t row = 0; row < disparityMap.rows; row++) {
		for (size_t col = 0; col < disparityMap.cols; col++) {
			flow.at<cv::Point2f>(row, col) = cv::Point2f(disparityMap.at<float>(row, col) - 00.0f, 0);
		}
	}
	return flow;
}

void stereo::Visualizer::MotionToColor(const cv::Mat &motion, 
	cv::Mat &colorMat, float maxmotion) {
	//CShape sh = motim.Shape();
	//int width = sh.width, height = sh.height;
	int width, height;
	width = motion.cols;
	height = motion.rows;
	//colim.ReAllocate(CShape(width, height, 3));
	colorMat.create(height, width, CV_8UC3);
	int x, y;
	// determine motion range:
	float maxx = -250, maxy = -250;
	float minx = 250, miny = 250;
	float maxrad = -1;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			float fy = motion.at<cv::Vec2f>(y, x)[1];
			float fx = motion.at<cv::Vec2f>(y, x)[0];
#ifdef _WIN32
			maxx = __max(maxx, fx);
			maxy = __max(maxy, fy);
			minx = __min(minx, fx);
			miny = __min(miny, fy);
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = __max(maxrad, rad);
#endif
#ifdef __linux__
			maxx = std::max<float>(maxx, fx);
			maxy = std::max<float>(maxy, fy);
			minx = std::min<float>(minx, fx);
			miny = std::min<float>(miny, fy);
			float rad = sqrt(fx * fx + fy * fy);
			maxrad = std::max<float>(maxrad, rad);
#endif
		}
	}
	if (maxmotion > 0) // i.e., specified on commandline
		maxrad = maxmotion;
	if (maxrad == 0) // if flow == 0 everywhere
		maxrad = 1;
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			float fy = motion.at<cv::Vec2f>(y, x)[1];
			float fx = motion.at<cv::Vec2f>(y, x)[0];
			uchar *pix = &colorMat.at<cv::Vec3b>(y, x)[0];
			computeColor(fx / maxrad, fy / maxrad, pix);
		}
	}
}


/**
@brief visualize a disparity map
@param cv::Mat disparityMap: input disparity map
@return cv::Mat visualImg
*/
cv::Mat stereo::Visualizer::visualize(cv::Mat disparityMap) {
	cv::Mat img;
	this->MotionToColor(disparityToFlow(disparityMap), img, 50);
	return img;
}