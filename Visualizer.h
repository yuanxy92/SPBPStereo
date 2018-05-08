/**
@brief stereo class for visualization
@author Shane Yuan
@date May 08, 2018
*/

#ifndef __SPBP_STEREO_VISUALIZER_H__ 
#define __SPBP_STEREO_VISUALIZER_H__

#include <cstdlib>
#include <cstdio>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>

namespace stereo {
	#define M_PI       3.14159265358979323846
	#define MAXCOLS 60
	class Visualizer {
	private:
		int ncols;
		int colorwheel[MAXCOLS][3];
	public:

	private:
		// functions to generate colors
		void makecolorwheel();
		void setcols(int r, int g, int b, int k);
		void computeColor(float fx, float fy, uchar *pix);
		void MotionToColor(const cv::Mat &motion,
			cv::Mat &colorMat, float maxmotion);
		// functions to convert display map to flow maps
		cv::Mat disparityToFlow(cv::Mat disparityMap);
	public:
		Visualizer();
		~Visualizer();
		
		/**
		@brief visualize a disparity map
		@param cv::Mat disparityMap: input disparity map
		@return cv::Mat visualImg
		*/
		cv::Mat visualize(cv::Mat disparityMap);
	};

}

#endif