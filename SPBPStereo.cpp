/**
@brief stereo class for super-pixel belief propagation
@author Shane Yuan
@date May 08, 2018
*/

#include "SPBPStereo.h"

stereo::SPBPStereo::SPBPStereo() {}
stereo::SPBPStereo::~SPBPStereo() {}

/**********************************************************************/
/*                          private functions                         */
/**********************************************************************/
/**
@brief randomize disparity map
@return int
*/
int stereo::SPBPStereo::randomDisparityMap() {
	// randomize 
	for (size_t k = 0; k < param.numOfK; k++) {
		for (size_t i = 0; i < leftSpGraphPtr->num; i++) {
			float dispar = (static_cast<float>(rand()) / RAND_MAX) * (param.maxDisparity
				- param.minDisparity) + param.minDisparity;
		}
	}
	return 0;
}

/**
@brief estimate depth image
@return int
*/
int stereo::SPBPStereo::estimate() {
	// create super pixels
	std::shared_ptr<stereo::SPGraph> spGraph = 
		spCreatorPtr->createSuperPixelGraph(leftImg);

	return 0;
}

/**********************************************************************/
/*                          public functions                          */
/**********************************************************************/
/**
@brief init superpixel belief propagation stereo
@param StereoParam param : input parameter for stereo
@return int
*/
int stereo::SPBPStereo::init(stereo::StereoParam param) {
	this->param = param;
	leftSpGraphPtr = std::make_shared<SPGraph>();
	spCreatorPtr = std::make_shared<SuperPixelGraph>();
	spCreatorPtr->init(stereo::SuperPixelParam());
	return 0;
}

/**
@brief estimate stereo from two images
@param cv::Mat leftImg: input image of left view
@param cv::Mat rightImg: input image of right view
@return int
*/
cv::Mat stereo::SPBPStereo::estimate(cv::Mat leftImg,
	cv::Mat rightImg) {
	this->leftImg = leftImg;
	this->rightImg = rightImg;
	estimate();
	return depthImg;
}