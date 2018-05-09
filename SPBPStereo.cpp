/**
@brief stereo class for super-pixel belief propagation
@author Shane Yuan
@date May 08, 2018
*/

#include "SPBPStereo.h"
#include <opencv2/ximgproc.hpp>

stereo::SPBPStereo::SPBPStereo() {}
stereo::SPBPStereo::~SPBPStereo() {}

/**********************************************************************/
/*                          private functions                         */
/**********************************************************************/
/**
@brief tansfer vec3b to vec3f
@param cv::Vec3b input: input vec3b value
@return cv::Vec3f: return vec3f value
*/
cv::Vec3f stereo::SPBPStereo::vec3bToVec3f(cv::Vec3b input) {
	cv::Vec3f val;
	val.val[0] = static_cast<float>(input.val[0]);
	val.val[1] = static_cast<float>(input.val[1]);
	val.val[2] = static_cast<float>(input.val[2]);
	return val;
}

/**
@brief calculate gradient image
@return int
*/
int stereo::SPBPStereo::calculateGradientImage() {
	cv::Mat leftGrayImg, rightGrayImg;
	cv::cvtColor(leftImg, leftGrayImg, cv::COLOR_BGR2GRAY);
	cv::cvtColor(rightImg, rightGrayImg, cv::COLOR_BGR2GRAY);
	std::vector<cv::Mat> dxy(2);
	cv::Sobel(leftGrayImg, dxy[0], leftGrayImg.depth(), 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel(leftGrayImg, dxy[1], leftGrayImg.depth(), 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
	cv::merge(dxy, leftGradImg);
	cv::Sobel(rightGrayImg, dxy[0], leftGrayImg.depth(), 1, 0, 3, 1, 0, cv::BORDER_REPLICATE);
	cv::Sobel(rightGrayImg, dxy[1], leftGrayImg.depth(), 0, 1, 3, 1, 0, cv::BORDER_REPLICATE);
	cv::merge(dxy, rightGradImg);
	return 0;
}

/**
@brief get local data cost per label
@param int spInd: index of super pixel
@param float dispar: input disparity to calculate disparity
@return cv::Mat_<float> dataCost
*/
cv::Mat_<float> stereo::SPBPStereo::getLocalDataCostPerLabel(int spInd, float dispar) {
	cv::Mat localDataCost;
	// get rectangle
	cv::Rect rect;
	rect.x = leftSpGraphPtr->nodes[spInd].rangeExpand.val[0];
	rect.y = leftSpGraphPtr->nodes[spInd].rangeExpand.val[1];
	rect.width = leftSpGraphPtr->nodes[spInd].rangeExpand.val[2] -
		leftSpGraphPtr->nodes[spInd].rangeExpand.val[0];
	rect.height = leftSpGraphPtr->nodes[spInd].rangeExpand.val[3] -
		leftSpGraphPtr->nodes[spInd].rangeExpand.val[1];
	// rgb color patch and gradient patch
	cv::Mat colorPatch, gradPatch;
	localDataCost.create(rect.size(), CV_32F);
	for (size_t row = 0; row < rect.height; row++) {
		for (size_t col = 0; col < rect.width; col++) {
			if (col + rect.x + dispar < rightImg.cols) {
				cv::Vec3b leftColorVal = leftImg.at<cv::Vec3b>(row + rect.y, col + rect.x);
				cv::Vec3b rightColorVal = rightImg.at<cv::Vec3b>(row + rect.y, col + rect.x + dispar);
				cv::Vec2b leftGradVal = leftGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x);
				cv::Vec2b rightGradVal = rightGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x + dispar);
				float colorCost = (abs(static_cast<float>(leftColorVal.val[0]) - static_cast<float>(rightColorVal.val[0]))
					+ abs(static_cast<float>(leftColorVal.val[1]) - static_cast<float>(rightColorVal.val[1]))
					+ abs(static_cast<float>(leftColorVal.val[2]) - static_cast<float>(rightColorVal.val[2]))) / 3;
				float gradCost = (abs(static_cast<float>(leftGradVal.val[0]) - static_cast<float>(rightGradVal.val[0]))
					+ abs(static_cast<float>(leftGradVal.val[1]) - static_cast<float>(rightGradVal.val[1]))) / 2;
				localDataCost.at<float>(row, col) = (1 - param.alpha) * std::min<float>(colorCost, 10)
				 	+ param.alpha * std::min<float>(gradCost, 2);
			}
			else {
				localDataCost.at<float>(row, col) = 255;
			}
		}
	}
	// aggregate cost using using guide filter
	leftImg(rect).copyTo(colorPatch);
	leftGradImg(rect).copyTo(gradPatch);
	// 
	cv::ximgproc::guidedFilter(colorPatch, localDataCost, localDataCost, 20, 4);
	return localDataCost;
}


/**
@brief randomize disparity map
@return int
*/
int stereo::SPBPStereo::randomDisparityMap() {
	dataCost_k.create(width * height, param.numOfK);
	label_k.create(width * height, param.numOfK);
	// randomize 
	for (size_t i = 0; i < leftSpGraphPtr->num; i++) { // i is super pixel index
		int k = 0;
		int repInd = leftSpGraphPtr->nodes[i].repInd;
		std::vector<float> labelVecs;
		// random top K depth
		while (k < param.numOfK) {
			float dispar = (static_cast<float>(rand()) / RAND_MAX) * (param.maxDisparity
				- param.minDisparity) + param.minDisparity;
			if (dispar >= param.minDisparity && dispar <= param.maxDisparity) {
				bool needReRand = false;
				for (int kinside = 0; kinside < k; kinside ++) {
					if (fabs(label_k[repInd][kinside] - dispar) <= 0.01)
						needReRand = true;
				}
				if (needReRand == false) {
					for (size_t pxInd = 0; pxInd < leftSpGraphPtr->nodes[i].pixels.size(); pxInd++) {
						label_k[leftSpGraphPtr->nodes[i].pixels[pxInd]][k] = dispar;
					}
					k++;
					labelVecs.push_back(dispar);
				}
			}
		}
		// calculate data cost
		std::vector<cv::Mat> costs(param.numOfK);
		for (size_t k = 0; k < costs.size(); k++) {
			costs[k] = this->getLocalDataCostPerLabel(i, labelVecs[k]);
		}
		// assign calculated data cost to dataCost_k
		int pt, px, py, lux, luy;
        for (int spPxInd = 0; spPxInd < leftSpGraphPtr->nodes[i].pixels.size(); spPxInd ++) {
            //cout<<ii<<endl;
            pt = leftSpGraphPtr->nodes[i].pixels[spPxInd];
            px = pt / width;
            py = pt % width;
			lux = leftSpGraphPtr->nodes[i].rangeExpand.val[0];
			luy = leftSpGraphPtr->nodes[i].rangeExpand.val[1];
            for (size_t k = 0; k < param.numOfK; k++) {
                dataCost_k[pt][k] = costs[k].at<float>(px - lux, py - luy);
            }
        }	
	}
	// init message for belief propagation
	message.create(width * height, 4);
	message.setTo(0);
	for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            cv::Vec3f centerVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j));
            cv::Vec3f upVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i - 1, j));
            cv::Vec3f downVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i + 1, j));
            cv::Vec3f leftVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j - 1));
            cv::Vec3f rightVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j + 1));
            smoothWeight[i][j][0] = param.lambda_smooth * std::exp(-cv::norm(centerVal - upVal) / 20.0f);
            smoothWeight[i][j][1] = param.lambda_smooth * std::exp(-cv::norm(centerVal - downVal) / 20.0f);
            smoothWeight[i][j][2] = param.lambda_smooth * std::exp(-cv::norm(centerVal - leftVal) / 20.0f);
            smoothWeight[i][j][3] = param.lambda_smooth * std::exp(-cv::norm(centerVal - rightVal) / 20.0f);
        }
    }
	return 0;
}

/**
@brief belief propagation variables
@return int
*/
int stereo::SPBPStereo::beliefPropagation() {

	return 0;
}

/**
@brief estimate depth image
@return int
*/
int stereo::SPBPStereo::estimate() {
	// calculate gradient image
	width = leftImg.cols;
	height = leftImg.rows;
	this->calculateGradientImage();
	// create super pixels
	leftSpGraphPtr = spCreatorPtr->createSuperPixelGraph(leftImg);
	// randomize disparity map
	this->randomDisparityMap();
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