//
//  main.cpp
//  HelloOpenCV
//
//  Created by Asaf Levy on 05/12/2018.
//  Copyright © 2018 Asaf Levy. All rights reserved.
//

#include <iostream>
#include <stdint.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

#include "utils.hpp"
#include "homography.h"
#include "Reconst.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;


int main(int argc, const char * argv[]) {

    // Load image:
    Mat left_image = imread("/Users/asaf/Desktop/left_110_2018-11-15-09-47-51-650.jpg", IMREAD_GRAYSCALE);
    Mat right_image = imread("/Users/asaf/Desktop/right_110_2018-11-15-09-47-51-650.jpg", IMREAD_GRAYSCALE);
    resize(left_image, left_image, Size(), 0.5, 0.5);
    resize(right_image, right_image, Size(), 0.5, 0.5);
    
    Mat warped, corr_intensities, corr_x_derivative, corr_y_derivative, std_map, cost_map, masked;

    // Some arrays for camera parameters:
    double KmatRef[3][3] = {{6811.10072821551/2, 0, 961.272247363782/2 },
        {0, 6824.78998611754/2, 647.340658150072/2},
        {0, 0, 1}};

    double Kmat[3][3] = {{6823.01473173781/2, 0, 967.153157981033/2 },
        {0, 6835.53522151543/2, 625.313551182732/2},
        {0, 0, 1}};

    double rotation[3][3] = {{0.999755456172702, -0.00604948034743692, -0.0212704405323284},
        {0.00606388484271891, 0.99998142673197, 0.000612773782206468},
        {0.0212663385077824,-0.000741605434102429, 0.999773570798835}};
    double translation[3][1] = {{-233.405463440145},
        {-1.47986087932354},
        {20.7515828935872}};

    double normal[1][3] = {0, 0, -1};

    double distance = 50000;

    // Create stereoParams object:
    stereoParams stereo_params;
    stereo_params.K = Mat(3, 3, CV_64FC1, Kmat);
    stereo_params.Kref = Mat(3, 3, CV_64FC1, KmatRef);
    stereo_params.r = Mat(3, 3, CV_64FC1, rotation);
    stereo_params.t = Mat(3, 1, CV_64FC1, translation);
    stereo_params.n = Mat(1, 3, CV_64FC1, normal);
    stereo_params.d = distance;

    // Create homography object:
    homography H(stereo_params);

    // Create x_map & y_map:
    Mat map_x, map_y;
    map_x.create(left_image.size(), CV_32FC1);
    map_y.create(left_image.size(), CV_32FC1);
    for(int r = 0; r < left_image.rows; r++)
    {
        for(int c = 0; c < left_image.cols; c++)
        {
            map_x.at<float>(r, c) = (H.getHomography().at<double>(0,0)*c + H.getHomography().at<double>(0,1)*r + H.getHomography().at<double>(0,2)) / (H.getHomography().at<double>(2,0)*c + H.getHomography().at<double>(2,1)*r + H.getHomography().at<double>(2,2));
            map_y.at<float>(r, c) = (H.getHomography().at<double>(1,0)*c + H.getHomography().at<double>(1,1)*r + H.getHomography().at<double>(1,2)) / (H.getHomography().at<double>(2,0)*c + H.getHomography().at<double>(2,1)*r + H.getHomography().at<double>(2,2));
        }
    }

    // Warp (loop starts from here):

    high_resolution_clock::time_point t1 = high_resolution_clock::now();   // Take time

    Reconst reconstruct(map_x, map_y, right_image, left_image, stereo_params);
    
    // Stop time:
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    cout <<"total time: " << duration << endl;
    
    reconstruct.getWarped(warped);
    reconstruct.getCorrelationIntensities(corr_intensities);
    reconstruct.getCorrelation_X_Derivative(corr_x_derivative);
    reconstruct.getCorrelation_Y_Derivative(corr_y_derivative);
    reconstruct.getStdMap(std_map);
    reconstruct.getCostMap(cost_map);
    reconstruct.getMasked(masked);
    
    // Show some results:
    imshow("warped", warped);
    waitKey(0);
    
    imshow("intense", corr_intensities);
    waitKey(0);
    
    imshow("x", corr_x_derivative);
    waitKey(0);
    
    imshow("y", corr_y_derivative);
    waitKey(0);
    
    imshow("std", std_map);
    waitKey(0);
    
    imshow("cost map", cost_map);
    waitKey(0);
    
    imshow("mask", masked);
    waitKey(0);
    
    return 0;

}
