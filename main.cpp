#include "SPBPStereo.h"

int main(int argc, char* argv[]) {
	cv::Mat img = cv::imread("E:\\data\\giga_stereo\\1\\data\\0\\0000.jpg");
	stereo::SuperPixelGraph graph;
	graph.init(stereo::SuperPixelParam());
	std::shared_ptr<stereo::SPGraph> spGraph = graph.createSuperPixelGraph(img);
	return 0;
}