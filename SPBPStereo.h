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
#include "CLMF/CLMF.h"

namespace stereo {

#define NUM_TOP_K 3
#define EPS 0.01

	// param for stereo estimation
	struct StereoParam {
		float minDisparity;
		float maxDisparity;
		int numOfK;
		float alpha;
		float lambda_smooth;
		int iterNum;
		float tau_s;
		StereoParam() {
			minDisparity = 0.0f;
			maxDisparity = 25.0f;
			numOfK = NUM_TOP_K;
			alpha = 0.9f;
			iterNum = 5;
			tau_s = 0.0f;
			lambda_smooth = 1.0f;
		}
	};

	// stereo class
	class SPBPStereo {
	private:
		// input images and output depth image
		StereoParam param;
		cv::Mat depthImg;
		cv::Mat leftImg;
		cv::Mat rightImg;
		cv::Mat_<float> disparityMap;
		// superpixel creator
		std::shared_ptr<SPGraph> leftSpGraphPtr;
		std::shared_ptr<SuperPixelGraph> spCreatorPtr;
		// visualizer
		std::shared_ptr<Visualizer> visualPtr;

		// variable used in belief propagation
		int width;
		int height;
		cv::Mat leftGradImg;
		cv::Mat rightGradImg;
		cv::Mat leftSmoothImg;
		cv::Mat rightSmoothImg;
		cv::Mat leftLBPImg;
		cv::Mat rightLBPImg;
		// data cost and correspondence label
		cv::Mat_<float> dataCost_k; // data cost
		cv::Mat_<float> label_k; // label, disparity
		// message used in belief propagation
		cv::Mat_<cv::Vec<float, NUM_TOP_K>> message;
		cv::Mat_<cv::Vec4f> smoothWeight;
		// guide map used for aggregate cost
		std::vector<int> hammingDist;
		std::shared_ptr<CFFilter> cfPtr;
		std::vector<cv::Mat_<cv::Vec4b>> guideMaps;

	public:

	private:
		/**
		@brief tansfer vec3b to vec3f
		@param cv::Vec3b input: input vec3b value
		@return cv::Vec3f: return vec3f value
		*/
		cv::Vec3f vec3bToVec3f(cv::Vec3b input);

		/**
		@brief feature calculation
		@param cv::Mat src: input image matrix
		@param cv::Mat dst: output lbp feature image
		@return int
		*/
		int calcLBPFeature(cv::Mat& src, cv::Mat& dst);

		/**
		@brief get hamming distance
		@param uchar feature1: first uchar feature 
		@param uchar feature2: first uchar feature
		@return int
		*/
		int getHammingDist(uchar feature1, uchar feature2);

		/**
		@brief estimate disparity map from data cost 
		@return int
		*/
		int estimateDisparity();

		/**
		@brief calculate gradient image
		@return int
		*/
		int calculateGradientImage();

		/**
		@brief get local data cost per label
		@param int spInd: index of super pixel
		@param float dispar: input disparity to calculate disparity
		@return cv::Mat_<float> localDataCost
		*/
		cv::Mat_<float> getLocalDataCostPerLabel(int spInd, float dispar);

		/**
		@brief function to modify arm length to fit sub image
		@return int
		*/
		int stereo::SPBPStereo::modifyCrossMapArmlengthToFitSubImage(const cv::Mat_<cv::Vec4b>& crMapIn,
			int maxArmLength, cv::Mat_<cv::Vec4b>& crMapOut);

		/**
		@brief randomize disparity map
		@return int
		*/
		int randomDisparityMap();

		/**
		@brief compute messages in belief propagation 
		@param const float* dis_belief: input dis-belief
		@param int p: input  
		*/
		float computeBPPerLabel(const float* dis_belief,
								int p,
								const float& disp_ref,
								float wt, float tau_s);

		/**
		@brief belief propagation variables
		@return int
		*/
		int beliefPropagation();

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
