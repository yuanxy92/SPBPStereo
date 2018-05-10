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
			ruler = 5.0f;
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
		cv::Vec4f range; // 0, 1, 2, 3 => min_x, min_y, max_x, max_y
		cv::Vec4f rangeExpand; // 0, 1, 2, 3 => min_x, min_y, max_x, max_y
	};

	struct SPGraph {
		size_t num; // int number of pixels
		cv::Mat label; // label mat
		std::vector<SPNode> nodes; // superpixel nodes in superpixel graph
	};

	struct NonLocalCandidate {
		float sumWeight;
		int spInd;
		NonLocalCandidate(float sw, int sp) : sumWeight(sw), spInd(sp) {}
		static bool larger(const NonLocalCandidate& ct1, const NonLocalCandidate& ct2) {
			return (ct1.sumWeight > ct2.sumWeight);
		}
	};

	class SuperPixelGraph {
	private:
		SuperPixelParam param;
	public:

	private:
		/**
		@brief tansfer vec3b to vec3f
		@param cv::Vec3b input: input vec3b value
		@return cv::Vec3f: return vec3f value
		*/
		cv::Vec3f vec3bToVec3f(cv::Vec3b input);

		/**
		@brief build non local graph to speed up belief propagation
		@param cv::Mat img: input image
		@param std::shared_ptr<SPGraph> spGraph: input/output superpixel graph
		@return int
		*/
		int buildNonLocalGraph(cv::Mat img, std::shared_ptr<SPGraph> spGraph);

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