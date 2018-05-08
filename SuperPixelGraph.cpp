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
	cv::Ptr<cv::ximgproc::SuperpixelSLIC> slic =
		cv::ximgproc::createSuperpixelSLIC(img, param.algorithm,
			param.regionSize, param.ruler);
	slic->iterate();
	slic->enforceLabelConnectivity(25);
	slic->getLabels(spGraph->label);
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
	// calculate expanded range
	int expand = 20;
	for (size_t i = 0; i < spGraph->nodes.size(); i++) {
		spGraph->nodes[i].rangeExpand.val[0] = std::max<float>(spGraph->nodes[i].range.val[0] - 20, 0);
		spGraph->nodes[i].rangeExpand.val[1] = std::max<float>(spGraph->nodes[i].range.val[1] - 20, 0);
		spGraph->nodes[i].rangeExpand.val[2] = std::min<float>(spGraph->nodes[i].range.val[2] + 20, img.cols);
		spGraph->nodes[i].rangeExpand.val[3] = std::min<float>(spGraph->nodes[i].range.val[3] + 20, img.rows);
	}
	// select representative pixels
	cv::RNG rng;
	for (size_t i = 0; i < spGraph->nodes.size(); i++) {
		spGraph->nodes[i].repInd = spGraph->nodes[i].pixels[rng.next() % spGraph->nodes[i].pixels.size()];
	}
	return spGraph;
}
