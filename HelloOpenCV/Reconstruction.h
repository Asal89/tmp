/*
 * Reconstruction.h
 *
 *  Created on: Nov 22, 2018
 *      Author: jetski
 */

#ifndef SRC_RECONSTRUCTION_H_
#define SRC_RECONSTRUCTION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"
#include "homography.h"
using namespace std;
using namespace cv;

class Reconstruction {

public:

	Reconstruction(stereoParams& str, homography& H, Mat& left, Mat& right);

	void getWarped(Mat& dst);

private:

	Mat warpedImage_;
};

#endif /* SRC_RECONSTRUCTION_H_ */
