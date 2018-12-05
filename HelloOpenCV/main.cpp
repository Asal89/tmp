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
#include "utils.hpp"
#include "homography.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    
    // Load image:
    Mat left_image = imread("/Users/asaf/Downloads/work/data/35mm/set1/35mm_pishpash_long/left/left_21_2018-11-15-09-47-30-172.jpg", IMREAD_GRAYSCALE);
    Mat right_image = imread("/Users/asaf/Downloads/work/data/35mm/set1/35mm_pishpash_long/right/right_21_2018-11-15-09-47-30-172.jpg", IMREAD_GRAYSCALE);
    
    // Some arrays for camera parameters:
    double KmatRef[3][3] = {{6811.10072821551, 0, 961.272247363782 },
        {0, 6824.78998611754, 647.340658150072},
        {0, 0, 1}};
    
    double Kmat[3][3] = {{6823.01473173781, 0, 967.153157981033 },
        {0, 6835.53522151543, 625.313551182732},
        {0, 0, 1}};
    
    double rotation[3][3] = {{0.999755456172702, -0.00604948034743692, -0.0212704405323284},
        {0.00606388484271891, 0.99998142673197, 0.000612773782206468},
        {0.0212663385077824,-0.000741605434102429, 0.999773570798835}};
    double translation[3][1] = {{-233.405463440145},
        {-1.47986087932354},
        {20.7515828935872}};
    
    double normal[1][3] = {0, 0, -1};
    
    double distance = 47000;
    
    // Create stereoParams object:
    stereoParams prm;
    prm.K = Mat(3, 3, CV_64FC1, Kmat);
    prm.Kref = Mat(3, 3, CV_64FC1, KmatRef);
    prm.r = Mat(3, 3, CV_64FC1, rotation);
    prm.t = Mat(3, 1, CV_64FC1, translation);
    prm.n = Mat(1, 3, CV_64FC1, normal);
    prm.d = distance;
    
    // Create homography object:
    homography H(prm);
    
    cout << "H =" << endl << " " << H.getHomography() << endl << endl;
    
    // Create x_map & y_map:
    Mat map_x, map_y, warped;
  
    warped.create(left_image.size(), left_image.type());
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
    
    // Warp image:
    remap(right_image, warped, map_x, map_y, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
    
    // Show results:
    Mat couple, img_small, warped_small;
    resize(left_image, img_small, Size(left_image.cols / 4, left_image.rows / 4));
    resize(warped, warped_small, Size(warped.cols / 4, warped.rows / 4));
    hconcat(img_small, warped_small, couple);
    imshow("output", couple);
    waitKey(0);
    
    // Show like imshowpair:
    Mat impair;
    addWeighted(img_small, 0.5, warped_small, 0.5, 0.0, impair);
    imshow("pair", impair);
    waitKey(0);
    
    
    return 0;
}
