//
//  Reconst.hpp
//  HelloOpenCV
//
//  Created by Asaf Levy on 06/12/2018.
//  Copyright © 2018 Asaf Levy. All rights reserved.
//

#ifndef Reconst_hpp
#define Reconst_hpp

#include <stdio.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"

using namespace std;
using namespace cv;

class Reconst{
public:
    Reconst(Mat& map_x, Mat& map_y, Mat& right_image, Mat& left_image, stereoParams& str);
    void getWarped(Mat& dst);
private:
    Mat warped_image_;
    Mat left_image_;
    stereoParams str_;
    
};
#endif /* Reconst_hpp */
