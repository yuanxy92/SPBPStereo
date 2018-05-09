/**
@brief stereo class for super-pixel graph construction
@author Shane Yuan
@date May 08, 2018
*/

#ifndef __SPBP_STEREO_SPGRAPH_H__ 
#define __SPBP_STEREO_SPGRAPH_H__

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>

namespace stereo {

	struct SuperPixelParam {
		int regionSize; // superpixel region size
		int algorithm; // algorithm used to create superpixels 
		float ruler; // parameter to enforce the smoothness of superpixels
		SuperPixelParam() {
			regionSize = sqrt(1200.0f * 1000.0f / 1000.0f);
			algorithm = cv::ximgproc::SLIC;
			ruler = 30.0f;
		}
		SuperPixelParam(int regionSize, int algorithm, float ruler) {
			this->regionSize = regionSize;
			this->algorithm = algorithm;
			this->ruler = ruler;
		}
	};

	struct SPNode {
		int index;
		int repInd; // representative pixel index
		std::vector<int> pixels; // pixel postions in superpixel
		std::set<int> adjs;
		cv::Vec4f range; // 0, 1, 2, 3 => min_x, min_y, min_z, min_w
		cv::Vec4f rangeExpand; // 0, 1, 2, 3 => min_x, min_y, min_z, min_w
	};

	struct SPGraph {
		size_t num; // int number of pixels
		cv::Mat label; // label mat
		std::vector<SPNode> nodes; // superpixel nodes in superpixel graph
	};

	class SuperPixelGraph {
	private:
		SuperPixelParam param;
	public:

	private:

	public:
		SuperPixelGraph();
		~SuperPixelGraph();

		/**
		@brief init superpixel graph class
		@param SuperPixelParam param: input parameter for superpixel graph
		@return int
		*/
		int init(SuperPixelParam param);

		/**
		@brief create superpixel graph
		@param cv::Mat img: input image to create superpixel graph
		@return std::shared_ptr<SPGraph>: superpixel graph
		*/
		std::shared_ptr<SPGraph> createSuperPixelGraph(cv::Mat img);
	};
}

#endif