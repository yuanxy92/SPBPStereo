/**
@brief stereo class for super-pixel graph construction
@author Shane Yuan
@date May 08, 2018
*/

#include "SuperPixelGraph.h"

#define _DEBUG_SUPERPIXEL_GRAPH

stereo::SuperPixelGraph::SuperPixelGraph() {}
stereo::SuperPixelGraph::~SuperPixelGraph() {}

/*************************************************************************/
/*                           private functions                           */
/*************************************************************************/
/**
@brief tansfer vec3b to vec3f
@param cv::Vec3b input: input vec3b value
@return cv::Vec3f: return vec3f value
*/
cv::Vec3f stereo::SuperPixelGraph::vec3bToVec3f(cv::Vec3b input) {
	cv::Vec3f val;
	val.val[0] = static_cast<float>(input.val[0]);
	val.val[1] = static_cast<float>(input.val[1]);
	val.val[2] = static_cast<float>(input.val[2]);
	return val;
}

/**
@brief build non local graph to speed up belief propagation
@param cv::Mat img: input image
@param std::shared_ptr<SPGraph> spGraph: input/output superpixel graph
@return int
*/
int stereo::SuperPixelGraph::buildNonLocalGraph(cv::Mat img, 
	std::shared_ptr<stereo::SPGraph> spGraph) {
	const float max_disparity = 40;
	const float sigmaSpatial = max_disparity * max_disparity;
	const float sigmaColor = 25.0 * 25.0;
	const float max_spatial_distance = max_disparity * max_disparity * 6.25;
	int spInd1, spInd2;
	int sampleNum = 30;
	int topk = 5;
	int width = img.cols;
	int height = img.rows;
	for (spInd1 = 0; spInd1 < spGraph->num; spInd1++) {
		std::vector<NonLocalCandidate> candVecs;
		candVecs.clear();
		int spIndSize1 = spGraph->nodes[spInd1].pixels.size();
		cv::Point2i p1, p2;
		for (spInd2 = 0; spInd2 < spGraph->num; spInd2++) {
			if (spInd1 == spInd2)
				continue;
			int spIndSize2 = spGraph->nodes[spInd2].pixels.size();
			float sumWeight = 0;
			int pxInd;
			for (pxInd = 0; pxInd < sampleNum; pxInd++) {
				int p1Ind = spGraph->nodes[spInd1].pixels[rand() % spIndSize1];
				p1.y = p1Ind / width;
				p1.x = p1Ind % width;
				int p2Ind = spGraph->nodes[spInd2].pixels[rand() % spIndSize2];
				p2.y = p2Ind / width;
				p2.x = p2Ind % width;

				float tmpSpatial = pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
				if (tmpSpatial > max_spatial_distance) {
					break;
				}

				cv::Vec3f pxVal1, pxVal2;
				pxVal1 = vec3bToVec3f(img.at<cv::Vec3b>(p1.y, p1.x));
				pxVal2 = vec3bToVec3f(img.at<cv::Vec3b>(p2.y, p2.x));
				float tmpColor = pow(pxVal1[0] - pxVal2[0], 2)
					+ pow(pxVal1[1] - pxVal2[1], 2)
					+ pow(pxVal1[2] - pxVal2[2], 2);
				float colorDis = exp(-tmpColor / sigmaColor);
				sumWeight += colorDis;
			}
			if (pxInd >= sampleNum)
				candVecs.push_back(NonLocalCandidate(sumWeight, spInd2));
		}

		std::sort(candVecs.begin(), candVecs.end(), NonLocalCandidate::larger);

		int canId = 0;
		int totalNeighbors = 0;
		for (canId = 0; canId < candVecs.size(); canId ++) {
			if (candVecs[canId].sumWeight < sampleNum * 0.2)
				break;
			int tmpId = candVecs[canId].spInd;
			// not itself
			if (tmpId != spInd1) {
				// not in its spatial adjacency list
				std::set<int>::iterator sIt;
				std::set<int>& sAdj = spGraph->nodes[spInd1].adjs;
				for (sIt = sAdj.begin(); sIt != sAdj.end(); sIt++) {
					if (tmpId == *sIt)
						break;
				}
				if (sIt == sAdj.end()) {
					spGraph->nodes[spInd1].adjs.insert(tmpId);
					spGraph->nodes[tmpId].adjs.insert(spInd1);
					if (totalNeighbors++ > topk)
						break;
				}
			}
		}
	}
	return 0;
}


/*************************************************************************/
/*                           public functions                            */
/*************************************************************************/
/**
@brief init superpixel graph class
@param SuperPixelParam param: input parameter for superpixel graph
@return int
*/
int stereo::SuperPixelGraph::init(SuperPixelParam param) {
	this->param = param;
	return 0;
}

/**
@brief create superpixel graph
@param cv::Mat img: input image to create superpixel graph
@return std::shared_ptr<SPGraph>: superpixel graph
*/
std::shared_ptr<stereo::SPGraph> stereo::SuperPixelGraph::createSuperPixelGraph(cv::Mat img) {
	std::shared_ptr<stereo::SPGraph> spGraph = std::make_shared<stereo::SPGraph>();
	// create superpixel
	param.regionSize = sqrt(img.rows * img.cols / 1500.0f);
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic =
		cv::ximgproc::createSuperpixelSLIC(img, param.algorithm,
			param.regionSize, param.ruler);
	slic->iterate();
	slic->enforceLabelConnectivity(40);
	slic->getLabels(spGraph->label);

	cv::Mat mask;
	slic->getLabelContourMask(mask);
	cv::Mat frame = img.clone();
	frame.setTo(cv::Scalar(0, 0, 255), mask);
	cv::imwrite("superpixel.png", frame);

	// make superpixel label continous
	std::set<int> labels;
	std::map<int, int> labelmaps;
	for (size_t row = 0; row < img.rows; row++) {
		for (size_t col = 0; col < img.cols; col++) {
			int label = spGraph->label.at<int>(row, col);
			labels.insert(label);
		}
	}
	spGraph->num = labels.size();
	spGraph->nodes.resize(spGraph->num);
	// init nodes in superpixel graph
	for (size_t i = 0; i < spGraph->num; i++) {
		spGraph->nodes[i].index = i;
		spGraph->nodes[i].range.val[0] = img.cols + 1;
		spGraph->nodes[i].range.val[1] = img.rows + 1;
		spGraph->nodes[i].range.val[2] = -1;
		spGraph->nodes[i].range.val[3] = -1;
	}
	// generate map by iterate set
	int ind = 0;
	for (auto &it : labels) {
		int label = it;
		labelmaps.insert(std::make_pair(it, ind));
		ind++;
	}
	// revise the label in label mat 
	for (size_t row = 0; row < img.rows; row++) {
		for (size_t col = 0; col < img.cols; col++) {
			int label = spGraph->label.at<int>(row, col);
			int newLabel = labelmaps.find(label)->second;
			spGraph->label.at<int>(row, col) = newLabel;
		}
	}
	// create grpah
	for (size_t row = 0; row < img.rows; row++) {
		for (size_t col = 0; col < img.cols; col++) {
			// get label 
			int label = spGraph->label.at<int>(row, col);
			spGraph->nodes[label].pixels.push_back(row * img.cols + col);
			// add adjs
			if (row < img.rows - 1) {
				int labelDown = spGraph->label.at<int>(row + 1, col);
				if (label != labelDown) {
					spGraph->nodes[label].adjs.insert(labelDown);
					spGraph->nodes[labelDown].adjs.insert(label);
				}
			}
			if (col < img.cols - 1) {
				int labelRight = spGraph->label.at<int>(row, col + 1);
				if (label != labelRight) {
					spGraph->nodes[label].adjs.insert(labelRight);
					spGraph->nodes[labelRight].adjs.insert(label);
				}
			}
			// update range
			if (col < spGraph->nodes[label].range.val[0])
				spGraph->nodes[label].range.val[0] = col;
			if (col > spGraph->nodes[label].range.val[2])
				spGraph->nodes[label].range.val[2] = col;
			if (row < spGraph->nodes[label].range.val[1])
				spGraph->nodes[label].range.val[1] = row;
			if (row > spGraph->nodes[label].range.val[3])
				spGraph->nodes[label].range.val[3] = row;
		}
	}
	// create non-local graph
	this->buildNonLocalGraph(img, spGraph);
	// calculate expanded range
	int expand = 20;
	for (size_t i = 0; i < spGraph->nodes.size(); i++) {
		spGraph->nodes[i].rangeExpand.val[0] = std::max<float>(spGraph->nodes[i].range.val[0] - 20, 0);
		spGraph->nodes[i].rangeExpand.val[1] = std::max<float>(spGraph->nodes[i].range.val[1] - 20, 0);
		spGraph->nodes[i].rangeExpand.val[2] = std::min<float>(spGraph->nodes[i].range.val[2] + 20, img.cols - 1);
		spGraph->nodes[i].rangeExpand.val[3] = std::min<float>(spGraph->nodes[i].range.val[3] + 20, img.rows - 1);
	}
	// select representative pixels
	cv::RNG rng;
	for (size_t i = 0; i < spGraph->nodes.size(); i++) {
		spGraph->nodes[i].repInd = spGraph->nodes[i].pixels[rng.next() % spGraph->nodes[i].pixels.size()];
	}
	return spGraph;
}
