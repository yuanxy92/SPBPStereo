/**
@brief stereo class for super-pixel belief propagation
@author Shane Yuan
@date May 08, 2018
*/

#ifndef __SPBP_STEREO_H__ 
#define __SPBP_STEREO_H__

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <memory>

#include <opencv2/opencv.hpp>

#include "SuperPixelGraph.h"
#include "Visualizer.h"

namespace stereo {

	// param for stereo estimation
	struct StereoParam {

	};

	// stereo class
	class SPBPStereo {
	private:
		// input images and output depth image
		StereoParam param;
		cv::Mat depthImg;
		cv::Mat leftImg;
		cv::Mat rightImg;
		cv::Mat disparityMap;
		// superpixel creator
		std::shared_ptr<SPGraph> leftSpGraphPtr;
		std::shared_ptr<SuperPixelGraph> spCreatorPtr;
		// visualizer
		std::shared_ptr<Visualizer> visualPtr;
	public:

	private:

		/**
		@brief estimate depth image
		@return int
		*/
		int estimate();

	public:
		SPBPStereo();
		~SPBPStereo();

		/**
		@brief init superpixel belief propagation stereo
		@param StereoParam param : input parameter for stereo
		@return int
		*/
		int init(StereoParam param);

		/**
		@brief estimate stereo from two images
		@param cv::Mat leftImg: input image of left view
		@param cv::Mat rightImg: input image of right view
		@return int
		*/
		cv::Mat estimate(cv::Mat leftImg, cv::Mat rightImg);
	};
}

#endif // __SPBP_STEREO_H__ 
