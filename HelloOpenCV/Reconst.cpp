//
//  Reconst.cpp
//  HelloOpenCV
//
//  Created by Asaf Levy on 06/12/2018.
//  Copyright © 2018 Asaf Levy. All rights reserved.
//

#include "Reconst.hpp"

using namespace std;
using namespace cv;

Reconst::Reconst(Mat& map_x, Mat& map_y, Mat& right_image, Mat& left_image, stereoParams& str){
    // Initialize private members:
    warped_image_.create(right_image.size(), right_image.type());
    remap(right_image, warped_image_, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
    left_image_ = left_image;
    str_ = str;
}

void Reconst::getWarped(Mat &dst){
    dst = warped_image_;
}
