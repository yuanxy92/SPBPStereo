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
			disparityMap[row][col] = label_k[ind][minInd];
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
			if (col + rect.x - dispar < rightImg.cols && col + rect.x - dispar >= 0) {
				cv::Vec3b leftColorVal = leftImg.at<cv::Vec3b>(row + rect.y, col + rect.x);
				cv::Vec3b rightColorVal = rightImg.at<cv::Vec3b>(row + rect.y, col + rect.x - dispar);
				cv::Vec2b leftGradVal = leftGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x);
				cv::Vec2b rightGradVal = rightGradImg.at<cv::Vec2b>(row + rect.y, col + rect.x - dispar);
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
	leftSmoothImg(rect).copyTo(colorPatch);
	cv::ximgproc::guidedFilter(colorPatch, localDataCost, localDataCost, 20, 8);
	//cv::GaussianBlur(localDataCost, localDataCost, cv::Size(21, 21), 12, 12, cv::BORDER_REPLICATE);
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
	message.setTo(0);
	smoothWeight.create(height, width);
	smoothWeight.setTo(cv::Scalar(10, 10, 10, 10));
	for (int i = 1; i < height - 1; ++i) {
        for (int j = 1; j < width - 1; ++j) {
            cv::Vec3f centerVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j));
            cv::Vec3f upVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i - 1, j));
            cv::Vec3f downVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i + 1, j));
            cv::Vec3f leftVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j - 1));
            cv::Vec3f rightVal = vec3bToVec3f(leftImg.at<cv::Vec3b>(i, j + 1));
            smoothWeight[i][j][0] = param.lambda_smooth * std::exp(-cv::norm(centerVal - leftVal) / 20.0f);
            smoothWeight[i][j][1] = param.lambda_smooth * std::exp(-cv::norm(centerVal - rightVal) / 20.0f);
            smoothWeight[i][j][2] = param.lambda_smooth * std::exp(-cv::norm(centerVal - upVal) / 20.0f);
            smoothWeight[i][j][3] = param.lambda_smooth * std::exp(-cv::norm(centerVal - downVal) / 20.0f);
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
		float cost_tp = dis_belief[k] + wt * std::min<float>(abs(disp_ref - label_k[p][k]), tau_s);
#if 0
		float cost_tp = dis_belief[k] + wt * std::min<float>((pow(disp_ref[0] - label_k[p][k], 2), tau_s);
#endif
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
	cv::waitKey(5);

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
			for (sIt = sAdj.begin(); sIt != sAdj.end(); ++sIt) {
				//repPixels1[*sIt] = superpixelsList1[*sIt][ind];
				leftSpGraphPtr->nodes[*sIt].repInd = leftSpGraphPtr->nodes[*sIt].
					pixels[rand() % leftSpGraphPtr->nodes[*sIt].pixels.size()];
				float test_label;
				for (int k = 0; k < NUM_TOP_K; k++) {
					//test_label= cur_label_k[k];
					test_label = label_k[leftSpGraphPtr->nodes[*sIt].repInd][k];
					bool isNew = true;
					for (size_t labelInd = 0; labelInd < vec_label_nei.size(); labelInd++) {
						if (abs(test_label - vec_label_nei[labelInd]) <= EPS) {
							isNew = false;
							break;
						}
					}
					if (isNew) {
						vec_label_nei.push_back(test_label);
					}
				}
			}
			// get labels from neighbor superpixels with random search
			leftSpGraphPtr->nodes[spInd].repInd = leftSpGraphPtr->nodes[spInd].
				pixels[rand() % leftSpGraphPtr->nodes[spInd].pixels.size()];
			for (size_t k = 0; k < NUM_TOP_K; k++) {
				float test_label = label_k[leftSpGraphPtr->nodes[spInd].repInd][k];
				float mag = (param.maxDisparity - param.minDisparity) / 8.0f;
				for (; mag >= 1.0f; mag /= 2.0f) {
					float test_label_random = test_label + ((float(rand()) / RAND_MAX) - 0.5) * 2.0 * mag;
					if (test_label_random >= param.minDisparity && test_label_random <= param.maxDisparity) {
						bool isNew = true;
						for (size_t labelInd = 0; labelInd < vec_label_nei.size(); labelInd++) {
							if (abs(test_label_random - vec_label_nei[labelInd]) <= EPS) {
								isNew = false;
								break;
							}
						}
						if (isNew) {
							vec_label_nei.push_back(test_label_random);
						}
					}
				}
			}

			// calculate data cost
			const int vec_size = vec_label_nei.size();
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
						vec_label.push_back(test_label);
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
					// update messages with propagation and random search labels
					float lux = leftSpGraphPtr->nodes[spInd].range.val[0];
					float luy = leftSpGraphPtr->nodes[spInd].range.val[1];
					for (int test_id = 0; test_id < vec_label_nei.size(); ++test_id) {
						float test_label = vec_label_nei[test_id];
						vec_label.push_back(test_label);
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

					// compute top k labels with most confident disparities
					size_t vec_in_size = vec_belief.size();
					for (int i = 0; i < NUM_TOP_K; i++) {
						float belief_min = vec_belief[i];
						int id = i;
						for (size_t j = i + 1; j < vec_in_size; j++) {
							if (vec_belief[j] < belief_min) {
								belief_min = vec_belief[j];
								id = j;
							}
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
						vec_belief[id] = vec_belief[i];
						if (!vec_mes_l.empty())
							vec_mes_l[id] = vec_mes_l[i];
						if (!vec_mes_r.empty())
							vec_mes_r[id] = vec_mes_r[i];
						if (!vec_mes_u.empty())
							vec_mes_u[id] = vec_mes_u[i];
						if (!vec_mes_d.empty())
							vec_mes_d[id] = vec_mes_d[i];
						vec_label[id] = vec_label[i];
						vec_d_cost[id] = vec_d_cost[i];
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
		cv::waitKey(5);
	}
	// diplay disparity map
	std::cout << cv::format("SPBP finished ...") << std::endl;
	this->estimateDisparity();
	display = this->visualPtr->visualize(disparityMap);
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
	cv::ximgproc::l0Smooth(leftImg, leftSmoothImg, 0.02, 2.0);
	cv::ximgproc::l0Smooth(rightImg, rightSmoothImg, 0.02, 2.0);
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