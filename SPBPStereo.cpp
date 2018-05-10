/**
@brief stereo class for super-pixel belief propagation
@author Shane Yuan
@date May 08, 2018
*/

#include "SPBPStereo.h"
#include <opencv2/ximgproc.hpp>

#define _DEBUG_SPBP

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
@brief push new elements into vector
@param std::vector<float> & vecs: input vectors
@param float val: input new value
@return int
*/
int stereo::SPBPStereo::pushNewElements(std::vector<float> & vecs, float val) {
	bool exist = false;
	for (size_t i = 0; i < vecs.size(); i++) {
		if (abs(val - vecs[i]) < EPS) {
			exist = true;
			return false;
		}
	}
	vecs.push_back(val);
	return true;
}

/**
@brief feature calculation
@param cv::Mat src: input image matrix
@param cv::Mat dst: output lbp feature image
@return int
*/
int stereo::SPBPStereo::calcLBPFeature(cv::Mat& src,
	std::vector<std::vector<std::bitset<CENSUS_WINDOW_SIZE
	* CENSUS_WINDOW_SIZE>>> & dst) {
	// change to gray image
	cv::Mat input;
	if (src.channels() > 1) 
		cv::cvtColor(src, input, cv::COLOR_BGR2GRAY);
	else input = src;
	// init vector
	dst.resize(src.rows);
	for (size_t i = 0; i < dst.size(); i++) {
		dst[i].resize(src.cols);
	}
	int gap = param.lbp_gap;
	// calculate census transform
	int hei_side = (CENSUS_WINDOW_SIZE - 1) / 2;
	int wid_side = (CENSUS_WINDOW_SIZE - 1) / 2;
	int tempValue;
	for (size_t row = 0; row < src.rows; row ++) {
		for (size_t col = 0; col < src.cols; col ++) {
			uchar centerValue = input.at<uchar>(row, col);
			int censusIdx = 0;
			for (int y = -hei_side * gap; y <= hei_side * gap; y = y + gap) {
				for (int x = -wid_side * gap; x <= wid_side * gap; x = x + gap) {
					if (row + y < 0 || row + y >= src.rows || col + x < 0 || col + x >= src.cols) {
						tempValue = centerValue;
					}
					else {
						tempValue = input.at<uchar>(row + y, col + x);
					}
					dst[row][col][censusIdx] = centerValue > tempValue ? 1 : 0;
					++censusIdx;
				}
			}
		}
	}
	
	return 0;
}

/**
@brief get hamming distance
@param uchar feature1: first uchar feature
@param uchar feature2: first uchar feature
@return int
*/
int stereo::SPBPStereo::getHammingDist(uchar feature1, uchar feature2) {
	uchar andFeature = feature1 ^ feature2;
	return hammingDist[static_cast<int>(andFeature)];
}

/**
@brief estimate disparity map from data cost
@return int
*/
int stereo::SPBPStereo::estimateDisparity() {
	float cost_perpixel[NUM_TOP_K];
	float tmp;
	//Mat_<Vec2f> flow_t(height1,width1);
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			int ind = row * width + col;
			for (int k = 0; k < NUM_TOP_K; k++)
				cost_perpixel[k] = dataCost_k[ind][k] + message[ind][0][k] +
				message[ind][1][k] + message[ind][2][k] + message[ind][3][k];

			float minCost = cost_perpixel[0];
			int minInd = 0;
			for (int k = 0; k < NUM_TOP_K; k++) {
				if (cost_perpixel[k] < minCost) {
					minCost = cost_perpixel[k];
					minInd = k;
				}
			}
			disparityMap[row][col] = label_k[ind][0];
		}
	}
	return 0;
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

//! popcount LUT for 8-bit vectors
static const uchar popcount_LUT8[256] = {
	0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
	4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

//! computes the population count of an N-byte vector using an 8-bit popcount LUT
template<typename T> static inline size_t popcount(T x) {
	size_t nBytes = sizeof(T);
	size_t nResult = 0;
	for (size_t l = 0; l<nBytes; ++l)
		nResult += popcount_LUT8[(uchar)(x >> l * 8)];
	return nResult;
}

//! computes the hamming distance between two N-byte vectors using an 8-bit popcount LUT
template<typename T> static inline size_t hdist(T a, T b) {
	return popcount(a^b);
}

/**
@brief get local data cost per label
@param int spInd: index of super pixel
@param float dispar: input disparity to calculate disparity
@return cv::Mat_<float> dataCost
*/
cv::Mat_<float> stereo::SPBPStereo::getLocalDataCostPerLabel(int spInd, float dispar) {
	cv::Mat_<float> localDataCost;
	// get rectangle
	cv::Rect rect;
	rect.x = leftSpGraphPtr->nodes[spInd].rangeExpand.val[0];
	rect.y = leftSpGraphPtr->nodes[spInd].rangeExpand.val[1];
	rect.width = leftSpGraphPtr->nodes[spInd].rangeExpand.val[2] -
		leftSpGraphPtr->nodes[spInd].rangeExpand.val[0] + 1;
	rect.height = leftSpGraphPtr->nodes[spInd].rangeExpand.val[3] -
		leftSpGraphPtr->nodes[spInd].rangeExpand.val[1] + 1;
	// rgb color patch and gradient patch
	cv::Mat colorPatch, gradPatch;
	localDataCost.create(rect.size());
	for (size_t row = 0; row < rect.height; row++) {
		for (size_t col = 0; col < rect.width; col++) {
			if (col + rect.x - dispar < rightImg.cols && col + rect.x - dispar >= 0) {
				cv::Vec3b leftColorVal = leftImg.at<cv::Vec3b>(row + rect.y, col + rect.x);
				cv::Vec3b rightColorVal = rightImg.at<cv::Vec3b>(row + rect.y, col + rect.x - dispar);
				cv::Vec2b leftGradVal = leftGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x);
				cv::Vec2b rightGradVal = rightGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x - dispar);
				float colorCost = (abs(static_cast<float>(leftColorVal.val[0]) - static_cast<float>(rightColorVal.val[0]))
					+ abs(static_cast<float>(leftColorVal.val[1]) - static_cast<float>(rightColorVal.val[1]))
					+ abs(static_cast<float>(leftColorVal.val[2]) - static_cast<float>(rightColorVal.val[2]))) / 3;
				//float gradCost = (abs(static_cast<float>(leftGradVal.val[0]) - static_cast<float>(rightGradVal.val[0]))
				//	+ abs(static_cast<float>(leftGradVal.val[1]) - static_cast<float>(rightGradVal.val[1]))) / 2;
				//localDataCost.at<float>(row, col) = (1 - param.alpha) * std::min<float>(colorCost, 10)
				// 	+ param.alpha * std::min<float>(gradCost, 2);
				std::bitset<CENSUS_WINDOW_SIZE * CENSUS_WINDOW_SIZE> leftLBP = leftLBPImg[row + rect.y][col + rect.x];
				std::bitset<CENSUS_WINDOW_SIZE * CENSUS_WINDOW_SIZE> rightLBP = rightLBPImg[row + rect.y][col + rect.x - dispar];
				float dist_css = expCensusDiffTable[(leftLBP ^ rightLBP).count()];
				float dist_ce = expColorDiffTable[static_cast<int>(colorCost)];
				localDataCost(row, col) = 255 * (dist_css + dist_ce);
			}
			else {
				localDataCost(row, col) = 0.0f;
			}
		}
	}
	leftImg(rect).copyTo(colorPatch);
	cv::ximgproc::guidedFilter(colorPatch, localDataCost, localDataCost, 9, 4); 
	//cfPtr->FastCLMF0FloatFilterPointer(guideMaps[spInd], localDataCost, localDataCost);
	return localDataCost;
}

/**
@brief function to modify arm length to fit sub image
@return int
*/
int stereo::SPBPStereo::modifyCrossMapArmlengthToFitSubImage(const cv::Mat_<cv::Vec4b>& crMapIn,
	int maxArmLength, cv::Mat_<cv::Vec4b>& crMapOut) {
	int iy, ix, height, width;
	height = crMapIn.rows;
	width = crMapIn.cols;
	crMapOut = crMapIn.clone();
	// up
	for (iy = 0; iy < std::min<int>(maxArmLength, height); ++iy) {
		for (ix = 0; ix < width; ++ix) {
			crMapOut[iy][ix][1] = std::min<int>(iy, crMapOut[iy][ix][1]);
		}
	}
	// down
	int ky = maxArmLength - 1;
	for (iy = height - maxArmLength; iy < height; ++iy) {
		if (iy < 0) {
			--ky;
			continue;
		}
		for (ix = 0; ix < width; ++ix) {
			crMapOut[iy][ix][3] = std::min<int>(ky, crMapOut[iy][ix][3]);
		}
		--ky;
	}
	// left
	for (iy = 0; iy < height; ++iy) {
		for (ix = 0; ix < std::min<int>(width, maxArmLength); ++ix) {
			crMapOut[iy][ix][0] = std::min<int>(ix, crMapOut[iy][ix][0]);
		}
	}
	// right
	int kx;
	for (iy = 0; iy < height; ++iy) {
		kx = maxArmLength - 1;
		for (ix = width - maxArmLength; ix < width; ++ix) {
			if (ix < 0) {
				--kx;
				continue;
			}
			crMapOut[iy][ix][2] = std::min<int>(kx, crMapOut[iy][ix][2]);
			--kx;
		}
	}
	return 0;
}

/**
@brief randomize disparity map
@return int
*/
int stereo::SPBPStereo::randomDisparityMap() {
	dataCost_k.create(width * height, param.numOfK);
	label_k.create(width * height, param.numOfK);
	// generate hamming dist
	for (int i = 0; i <= CENSUS_WINDOW_SIZE * CENSUS_WINDOW_SIZE + 1; i ++)
		expCensusDiffTable[i] = 1.0 - exp(-i/30.0f);
	for (int i = 0; i < 256; i ++)
		expColorDiffTable[i] = 1.0 - exp(-i/60.0f);
	// randomize 
	for (size_t i = 0; i < leftSpGraphPtr->num; i++) { // i is super pixel index
		int k = 0;
		int repInd = leftSpGraphPtr->nodes[i].repInd;
		std::vector<float> labelVecs;
		// init guide filter
		int crossColorTau = 25;
		int crossArmLength = 9;
		// calculate sub-image and sub-crossmap
		cv::Mat leftBlurImg;
		cv::Mat leftImgf;
		cv::Mat_<cv::Vec4b> crossMap;
		leftImg.convertTo(leftImgf, CV_32FC3);
		cv::medianBlur(leftImgf, leftBlurImg, 3);
		cfPtr = std::make_shared<CFFilter>();
		cfPtr->GetCrossUsingSlidingWindow(leftBlurImg, crossMap, crossArmLength, crossColorTau);
		guideMaps.resize(leftSpGraphPtr->num);
		for (size_t spInd = 0; spInd < leftSpGraphPtr->num; spInd++) {
			int pxInd = leftSpGraphPtr->nodes[spInd].repInd;
			// extract sub-image from subrange
			int w = leftSpGraphPtr->nodes[spInd].rangeExpand.val[2] -
				leftSpGraphPtr->nodes[spInd].rangeExpand.val[0] + 1;
			int h = leftSpGraphPtr->nodes[spInd].rangeExpand.val[3] -
				leftSpGraphPtr->nodes[spInd].rangeExpand.val[1] + 1;
			int x = leftSpGraphPtr->nodes[spInd].rangeExpand.val[0];
			int y = leftSpGraphPtr->nodes[spInd].rangeExpand.val[1];

			cv::Mat_<cv::Vec4b> tmpCr;
			modifyCrossMapArmlengthToFitSubImage(crossMap(cv::Rect(x, y, w, h)), crossArmLength, tmpCr);
			guideMaps[spInd] = tmpCr.clone();
		}

		// random top K depth
		while (k < param.numOfK) {
			float dispar = floor((static_cast<float>(rand()) / RAND_MAX) * (param.maxDisparity
				- param.minDisparity) + param.minDisparity);
			if (dispar >= param.minDisparity && dispar <= param.maxDisparity) {
				bool needReRand = false;
				for (int kinside = 0; kinside < k; kinside ++) {
					if (fabs(label_k[repInd][kinside] - dispar) <= EPS)
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
            py = pt / width;
            px = pt % width;
			lux = leftSpGraphPtr->nodes[i].rangeExpand.val[0];
			luy = leftSpGraphPtr->nodes[i].rangeExpand.val[1];
            for (size_t k = 0; k < param.numOfK; k++) {
                dataCost_k[pt][k] = costs[k].at<float>(py - luy, px - lux);
            }
        }	
	}
	// init message for belief propagation
	message.create(width * height, 4);
	message.setTo(cv::Scalar(0, 0, 0));
	smoothWeight.create(height, width);
	smoothWeight.setTo(cv::Scalar(param.lambda_smooth, param.lambda_smooth,
		param.lambda_smooth, param.lambda_smooth));
	for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cv::Vec3f centerVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j));
			if (i > 0) {
				cv::Vec3f upVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i - 1, j));
				smoothWeight[i][j][2] = param.lambda_smooth * std::exp(-cv::norm(centerVal - upVal) / 20.0f);
			}
			if (i < height - 1) {
				cv::Vec3f downVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i + 1, j));
				smoothWeight[i][j][3] = param.lambda_smooth * std::exp(-cv::norm(centerVal - downVal) / 20.0f);
			}
			if (j > 0) {
				cv::Vec3f leftVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j - 1));
				smoothWeight[i][j][0] = param.lambda_smooth * std::exp(-cv::norm(centerVal - leftVal) / 20.0f);
			}
			if (j < width - 1) {
				cv::Vec3f rightVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j + 1));
				smoothWeight[i][j][1] = param.lambda_smooth * std::exp(-cv::norm(centerVal - rightVal) / 20.0f);
			}
        }
    }
	disparityMap.create(height, width);
	return 0;
}

/**
@brief compute messages in belief propagation
@param const float* dis_belief: input dis-belief
@param int p: input
*/
float stereo::SPBPStereo::computeBPPerLabel(const float* dis_belief,
	int p, const float& disp_ref, float wt, float tau_s) {
	float min_cost = 1e5;
	for (int k = 0; k < NUM_TOP_K; ++k) {
		float label_val = label_k[p][k];
		float cost_tp = dis_belief[k] + wt * std::min<float>(pow(disp_ref - label_val, 2), tau_s);
		if (cost_tp < min_cost)
			min_cost = cost_tp;
	}
	return min_cost;
}

/**
@brief belief propagation variables
@return int
*/
int stereo::SPBPStereo::beliefPropagation() {
	// dis-belief buffer
	float disBelief_l[NUM_TOP_K];
	float disBelief_r[NUM_TOP_K];
	float disBelief_u[NUM_TOP_K];
	float disBelief_d[NUM_TOP_K];
	const int buffersize = NUM_TOP_K * 50;
	std::vector<float> vec_label(buffersize), vec_label_nei(buffersize);
	std::vector<float> vec_mes_l(buffersize),
		vec_mes_r(buffersize),
		vec_mes_u(buffersize),
		vec_mes_d(buffersize),
		vec_belief(buffersize),
		vec_d_cost(buffersize);
	cv::Mat_<float> DataCost_nei;
	// diplay disparity map
	this->estimateDisparity();
	cv::Mat display = this->visualPtr->visualize(disparityMap);
	cv::imshow("display", display);
	cv::imwrite("visual_init.png", display);
	cv::waitKey(20);
	// start belief propagation
	int spBegin, spEnd, spStep;
	for (size_t iter = 0; iter < param.iterNum; iter++) {
		// clear neighbor lable vector
		if (iter % 2) {
			spBegin = leftSpGraphPtr->num - 1, spEnd = -1, spStep = -1;
		}
		else {
			spBegin = 0, spEnd = leftSpGraphPtr->num, spStep = 1;
		}
		// go through all the super pixels
		for (int spInd = spBegin; spInd != spEnd; spInd += spStep) {
			//std::cout << cv::format("Current processing super-pixel index: %d ...", spInd) << std::endl;
			vec_label_nei.clear();
			// get neighbor label vector
			std::set<int>::iterator sIt;
			std::set<int>& sAdj = leftSpGraphPtr->nodes[spInd].adjs;
			int ind = 0;
			// get labels from neighbor superpixels
			for (sIt = sAdj.begin(); sIt != sAdj.end(); sIt ++ ) {
				//repPixels1[*sIt] = superpixelsList1[*sIt][ind];
				leftSpGraphPtr->nodes[*sIt].repInd = leftSpGraphPtr->nodes[*sIt].
					pixels[rand() % leftSpGraphPtr->nodes[*sIt].pixels.size()];
				float test_label;
				for (int k = 0; k < NUM_TOP_K; k++) {
					test_label = label_k[leftSpGraphPtr->nodes[*sIt].repInd][k];
					pushNewElements(vec_label_nei, test_label);
				}
			}

			// calculate data cost
			const int vec_size = vec_label_nei.size();

#ifdef _DEBUG_SPBP1
			std::cout << cv::format("Superpixel %d, neighbor labels: %f, %f, %f\n", 
				spInd, label_k[leftSpGraphPtr->nodes[spInd].repInd][0], 
				label_k[leftSpGraphPtr->nodes[spInd].repInd][1],
				label_k[leftSpGraphPtr->nodes[spInd].repInd][2]);
			for (size_t i = 0; i < vec_size; i++)
				std::cout << vec_label_nei[i] << "\t";
			std::cout << std::endl;
#endif

			std::vector<cv::Mat_<float>> dataCost_nei(vec_size);
			for (size_t neiInd = 0; neiInd < vec_size; neiInd++) {
				dataCost_nei[neiInd] = getLocalDataCostPerLabel(spInd, vec_label_nei[neiInd]);
			}
			// init belief propagation
			cv::Vec4i curRange_s = leftSpGraphPtr->nodes[spInd].range;
			// init super pixel range
			int spy_s, spy_e, spx_s, spx_e, spx_step, spy_step;
			if (iter % 4 == 0) {
				spx_s = curRange_s[0];
				spx_e = curRange_s[2] + 1;
				spy_s = curRange_s[1];
				spy_e = curRange_s[3] + 1;
				spx_step = 1;
				spy_step = 1;
			}
			if (iter % 4 == 1) {
				spx_s = curRange_s[2];
				spx_e = curRange_s[0] - 1;
				spy_s = curRange_s[3];
				spy_e = curRange_s[1] - 1;
				spx_step = -1;
				spy_step = -1;
			}
			else if (iter % 4 == 2) {
				spx_s = curRange_s[2];
				spx_e = curRange_s[0] - 1;
				spx_step = -1;
				spy_s = curRange_s[1];
				spy_e = curRange_s[3] + 1;
				spy_step = 1;
			}
			else if (iter % 4 == 3)
			{
				spx_s = curRange_s[0];
				spx_e = curRange_s[2] + 1;
				spx_step = 1;
				spy_s = curRange_s[3];
				spy_e = curRange_s[1] - 1;
				spy_step = -1;
			}
			// go through all the pixels
			for (int by = spy_s; by != spy_e; by += spy_step) {
				for (int bx = spx_s; bx != spx_e; bx += spx_step) {
					// pixel positions
					int pcenter = by * width + bx;
					int pl = by * width + (bx - 1);
					int pu = (by - 1) * width + bx;
					int pr = by * width + (bx + 1);
					int pd = (by + 1) * width + bx;
					// compute dis-belief
					for (int k = 0; k < NUM_TOP_K; ++k) {
						if (bx != 0)
							disBelief_l[k] = message[pl][0][k] + message[pl][2][k] + 
							message[pl][3][k] + dataCost_k[pl][k];
						if (bx != width - 1)
							disBelief_r[k] = message[pr][1][k] + message[pr][2][k] + 
							message[pr][3][k] + dataCost_k[pr][k];
						if (by != 0)
							disBelief_u[k] = message[pu][0][k] + message[pu][1][k] + 
							message[pu][2][k] + dataCost_k[pu][k];
						if (by != height - 1)
							disBelief_d[k] = message[pd][0][k] + message[pd][1][k] + 
							message[pd][3][k] + dataCost_k[pd][k];
					}
					vec_label.clear();
					vec_mes_l.clear();
					vec_mes_r.clear();
					vec_mes_u.clear();
					vec_mes_d.clear();
					vec_belief.clear();
					vec_d_cost.clear();
					// update messages with current reference pixel's labels
					cv::Vec4f wt_s = smoothWeight[by][bx];
					for (int k = 0; k < NUM_TOP_K; ++k) {
						float test_label = label_k[pcenter][k];
						if (pushNewElements(vec_label, test_label)) {
							float dcost = dataCost_k[pcenter][k];
							vec_d_cost.push_back(dcost);
							float _mes_l = 0, _mes_r = 0, _mes_u = 0, _mes_d = 0;
							if (bx != 0) {
								_mes_l = computeBPPerLabel(disBelief_l, pl, test_label, wt_s[0], param.tau_s);
								vec_mes_l.push_back(_mes_l);
							}
							if (bx != width - 1) {
								_mes_r = computeBPPerLabel(disBelief_r, pr, test_label, wt_s[1], param.tau_s);
								vec_mes_r.push_back(_mes_r);
							}
							if (by != 0) {
								_mes_u = computeBPPerLabel(disBelief_u, pu, test_label, wt_s[2], param.tau_s);
								vec_mes_u.push_back(_mes_u);
							}
							if (by != height - 1) {
								_mes_d = computeBPPerLabel(disBelief_d, pd, test_label, wt_s[3], param.tau_s);
								vec_mes_d.push_back(_mes_d);
							}
							vec_belief.push_back(_mes_l + _mes_r + _mes_u + _mes_d + dcost);
						}
					}
					// update messages with propagation and random search labels
					float lux = leftSpGraphPtr->nodes[spInd].rangeExpand.val[0];
					float luy = leftSpGraphPtr->nodes[spInd].rangeExpand.val[1];
					for (int test_id = 0; test_id < vec_label_nei.size(); ++test_id) {
						float test_label = vec_label_nei[test_id];
						if (pushNewElements(vec_label, test_label)) {
							const cv::Mat_<float>& local = dataCost_nei[test_id];
							float dcost = dataCost_nei[test_id].at<float>(by - luy, bx - lux);
							vec_d_cost.push_back(dcost);
							//start_disp = clock();
							float _mes_l = 0, _mes_r = 0, _mes_u = 0, _mes_d = 0;
							if (bx != 0) {
								_mes_l = computeBPPerLabel(disBelief_l, pl, test_label, wt_s[0], param.tau_s);
								vec_mes_l.push_back(_mes_l);
							}
							if (bx != width - 1) {
								_mes_r = computeBPPerLabel(disBelief_r, pr, test_label, wt_s[1], param.tau_s);
								vec_mes_r.push_back(_mes_r);
							}
							if (by != 0) {
								_mes_u = computeBPPerLabel(disBelief_u, pu, test_label, wt_s[2], param.tau_s);
								vec_mes_u.push_back(_mes_u);
							}
							if (by != height - 1) {
								_mes_d = computeBPPerLabel(disBelief_d, pd, test_label, wt_s[3], param.tau_s);
								vec_mes_d.push_back(_mes_d);
							}
							vec_belief.push_back(_mes_l + _mes_r + _mes_u + _mes_d + dcost);
						}
					}
					// compute top k labels with most confident disparities
					size_t vec_in_size = vec_belief.size();
					int id;
					for (int i = 0; i < NUM_TOP_K; i++) {
						float belief_min = 999999999;
						float belief_max = 0;
						for (size_t j = 0; j < vec_in_size; j++) {
							if (vec_belief[j] < belief_min) {
								belief_min = vec_belief[j];
								id = j;
							}
							if (vec_belief[j] > belief_max)
								belief_max = vec_belief[j];
						}
						if (!vec_mes_l.empty())
							message[pcenter][0][i] = vec_mes_l[id];
						if (!vec_mes_r.empty())
							message[pcenter][1][i] = vec_mes_r[id];
						if (!vec_mes_u.empty())
							message[pcenter][2][i] = vec_mes_u[id];
						if (!vec_mes_d.empty())
							message[pcenter][3][i] = vec_mes_d[id];
						label_k[pcenter][i] = vec_label[id];
						dataCost_k[pcenter][i] = vec_d_cost[id];

						vec_belief[id] = belief_max + 1;
						belief_max++;
#ifdef _DEBUG_SPBP1		
						std::cout << cv::format("Superpixel %d, k = %d, choose lable = %f\n",
							spInd, i, vec_label[id]) << std::endl;
#endif
					}
					// message normalization
					for (int i = 0; i < 4; i++) {
						float val = 0.0;
						// TODO: How many elements are there in mes_pixel? NUM_TOP_K?
						// If that is the case we can just sum over vector to get val
						// We can also do loop unrolling directly
						for (int k = 0; k < NUM_TOP_K; k++)
							val += message[pcenter][i][k];
						val /= (float)NUM_TOP_K;
						for (int k = 0; k < NUM_TOP_K; k++)
							message[pcenter][i][k] -= val;
					}
				}
			}
		}
		// diplay disparity map
		std::cout << cv::format("Iteration %d, finished ...", iter) << std::endl;
		this->estimateDisparity();
		cv::Mat display = this->visualPtr->visualize(disparityMap);
		cv::imshow("display", display);
		cv::waitKey(20);
	}
	// diplay disparity map
	std::cout << cv::format("SPBP finished ...") << std::endl;
	this->estimateDisparity();
	cv::imwrite("visual0.png", disparityMap * 5);
	display = this->visualPtr->visualize(disparityMap);
	cv::imwrite("visual.png", display);
	cv::imshow("display", display);
	cv::waitKey(0);
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
	//cv::ximgproc::l0Smooth(leftImg, leftSmoothImg, 0.02, 2.0);
	//cv::ximgproc::l0Smooth(rightImg, rightSmoothImg, 0.02, 2.0);
	leftSmoothImg = leftImg;
	rightSmoothImg = rightImg;
	calcLBPFeature(leftImg, leftLBPImg);
	calcLBPFeature(rightImg, rightLBPImg);

	this->calculateGradientImage();
	// create super pixels
	leftSpGraphPtr = spCreatorPtr->createSuperPixelGraph(leftImg);
	// randomize disparity map
	this->randomDisparityMap();
	// belief propagation
	this->beliefPropagation();
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
	visualPtr = std::make_shared<Visualizer>();
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