/*
 * Reconstruction.cpp
 *
 *  Created on: Nov 22, 2018
 *      Author: jetski
 */

#include "Reconstruction.h"

Reconstruction::Reconstruction(stereoParams& str, homography& H, Mat& left, Mat& right){


}

void Reconstruction::getWarped(Mat& dst){
		dst = warpedImage_;
	};
