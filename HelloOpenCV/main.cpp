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

void derivative(Mat& src, Mat& dst, string axis)
{
    
    if (axis == "x")
    {
        double data[1][3] = {-1, 0, 1};
        Mat kernel(1, 3, CV_64FC1, data);
        filter2D(src, dst, -1, kernel, Point(0, 0), 0, BORDER_CONSTANT);
        return;
    }
    else if (axis == "y")
    {
        double data[3][1] = {{-1}, {0}, {1}};
        Mat kernel(3, 1, CV_64FC1, data);
        filter2D(src, dst, -1, kernel, Point(0, 0), 0, BORDER_CONSTANT);
        return;
    }
    else
    {
        cout << "Wrong Argument. pass 'x' or 'y' only." << endl;
        throw;
        return;
    }
}

void conv2(Mat& src, Mat& dst, int kernel_size)
{
    Mat kernel = Mat::ones(kernel_size, kernel_size, CV_64FC1)/ (double)(kernel_size*kernel_size);
    filter2D(src, dst, -1 , kernel, Point(0, 0), 0, BORDER_CONSTANT );
}

void positive(Mat& src, Mat& dst){
    dst = cv::max(src, 0);
}

void corr (Mat& f, Mat& g, int kernel_size, Mat& dst)
{
    Mat double_f, double_g, mean_f, mean_g, sqr_f, sqr_g, mean_sqr_f, mean_sqr_g, var_f, var_g, pos_var_f, pos_var_g, std_f, std_g, f_g, meanf_meamg, mean_fg, std_prod;
    double threshold = 0.5;
    
    f.convertTo(double_f, CV_64FC1);
    g.convertTo(double_g, CV_64FC1);
    
    conv2(double_f, mean_f, kernel_size);
    conv2(double_g, mean_g, kernel_size);
    
    sqr_f = double_f.mul(double_f);
    sqr_g = double_g.mul(double_g);
    
    conv2(sqr_f, mean_sqr_f, kernel_size);
    conv2(sqr_g, mean_sqr_g, kernel_size);
    
    var_f = mean_sqr_f - mean_f.mul(mean_f);
    var_g = mean_sqr_g - mean_g.mul(mean_g);
    
    positive(var_f, pos_var_f);
    positive(var_g, pos_var_g);
    
    sqrt(pos_var_f + (1 - pos_var_f) * threshold / 10, std_f);
    sqrt(pos_var_f + (1 - pos_var_g) * threshold / 10, std_g);
    
    f_g = double_f.mul(double_g);
    meanf_meamg = mean_f.mul(mean_g);
    conv2(f_g, mean_fg, kernel_size);
    std_prod = std_f.mul(std_g);
    dst = (mean_fg - meanf_meamg) / std_prod;
    return;
}

int main(int argc, const char * argv[]) {
    
    // Load image:
    Mat left_image = imread("/Users/asaf/Desktop/left_110_2018-11-15-09-47-51-650.jpg", IMREAD_GRAYSCALE);
    Mat right_image = imread("/Users/asaf/Desktop/right_110_2018-11-15-09-47-51-650.jpg", IMREAD_GRAYSCALE);
    Mat corr_intensities, corr_x_derivative, corr_y_derivative;
    
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

    // Create x_map & y_map:
    Mat map_x, map_y, warped;
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

    // Warp:
    Reconst rec(map_x, map_y, right_image, left_image, prm);
    rec.getWarped(warped);
    
    // Calculate derivatives:
    Mat left_x_derivative, left_y_derivative;
    Mat warped_x_derivative, warped_y_derivative;
    derivative(left_image, left_x_derivative, "x");
    derivative(left_image, left_y_derivative, "y");
    derivative(warped, warped_x_derivative, "x");
    derivative(warped, warped_y_derivative, "y");

    // Take time:
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    // Calculate correlation of intensities:
    corr(left_image, warped, 10, corr_intensities);

    // Calculate correlation of 'x' derivative:
    corr(left_x_derivative, warped_x_derivative, 10, corr_x_derivative);
    
    // Calculate correlation of 'y' derivative:
    corr(left_y_derivative, warped_y_derivative, 10, corr_y_derivative);
    
    // Stop time:
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>( t2 - t1 ).count();
    cout <<"time: " << duration << endl;
    
    imshow("intense", corr_intensities);
    waitKey(0);
    
    imshow("x", corr_x_derivative);
    waitKey(0);
    
    imshow("y", corr_y_derivative);
    waitKey(0);
    
    return 0;

}
