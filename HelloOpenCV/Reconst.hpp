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
    
    Reconst(Mat& map_x, Mat& map_y, Mat& right_image, Mat& left_image, stereoParams& stereo_params);
   
    void getWarped(Mat& dst);
    
    void getCorrelationIntensities(Mat& dst);
    
    void getCorrelation_X_Derivative(Mat& dst);
    
    void getCorrelation_Y_Derivative(Mat& dst);
    
    void getStdMap(Mat& dst);
    
    void getCostMap(Mat& dst);
    
    void getMasked(Mat& dst);
    
    
private:
    stereoParams stereoParams_;
    Mat leftImage_;
    Mat rightImage;
    Mat mapX_;
    Mat mapY_;
    Mat warped_image_;
    Mat left_x_derivative_;
    Mat left_y_derivative_;
    Mat warped_x_derivative_;
    Mat warped_y_derivative_;
    Mat corr_intensities_;
    Mat corr_x_derivative_;
    Mat corr_y_derivative_;
    Mat std_map_;
    Mat cost_map_;
    Mat masked_;
    
};
#endif /* Reconst_hpp */
