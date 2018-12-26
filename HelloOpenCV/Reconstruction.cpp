//
//  Reconstruction.cpp
//  HelloOpenCV
//
//  Created by Asaf Levy on 26/12/2018.
//  Copyright © 2018 Asaf Levy. All rights reserved.
//

#include "Reconstruction.hpp"
using namespace std;
using namespace cv;

/* Some helper functions: */
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
    Mat temp;
    Mat col_kernel = Mat::ones(kernel_size, 1, CV_64FC1)/ (double)(kernel_size);
    Mat row_kernel = Mat::ones(1, kernel_size, CV_64FC1)/ (double)(kernel_size);
    filter2D(src, temp, -1 , col_kernel, Point(0, 0), 0, BORDER_CONSTANT );
    filter2D(temp, dst, -1 , row_kernel, Point(0, 0), 0, BORDER_CONSTANT );
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
    sqrt(pos_var_g + (1 - pos_var_g) * threshold / 10, std_g);
    
    f_g = double_f.mul(double_g);
    meanf_meamg = mean_f.mul(mean_g);
    conv2(f_g, mean_fg, kernel_size);
    std_prod = std_f.mul(std_g);
    dst = (mean_fg - meanf_meamg) / std_prod;
    
    return;
}

void variance(Mat& corr_intensities, Mat& corr_x_derivative, Mat& corr_y_derivative, Mat& dst)
{
    Mat avarege = (1/(double)3) * (corr_intensities + corr_x_derivative + corr_y_derivative);
    
    // Standardize:
    Mat corr_intensities_shifted = corr_intensities - avarege;
    Mat corr_x_derivative__shifted = corr_x_derivative - avarege;
    Mat corr_y_derivative_shifted = corr_y_derivative - avarege;
    
    corr_intensities_shifted = corr_intensities_shifted.mul(corr_intensities_shifted);
    corr_x_derivative__shifted = corr_x_derivative__shifted.mul(corr_x_derivative__shifted);
    corr_y_derivative_shifted = corr_y_derivative_shifted.mul(corr_y_derivative_shifted);
    
    sqrt((1/(double)3) * (corr_intensities_shifted + corr_x_derivative__shifted + corr_y_derivative_shifted), dst);
}

void cost_function(Mat& corr_intensities, Mat& corr_x_derivative, Mat& corr_y_derivative, Mat& std_map, Mat& dst)
{
    double intensity_factor    = 1 ;
    double x_derivative_factor = 1 ;
    double y_derivative_factor = 1 ;
    double std_map_factor      = -8;
    
    dst = (intensity_factor * corr_intensities) + (x_derivative_factor * corr_x_derivative) + (y_derivative_factor * corr_y_derivative) + (std_map_factor * std_map);
    
}

void threshold_filter(Mat& cost_map, double threshold, Mat& masked, Mat& mask)
{
    
    double min, max;
    minMaxLoc(cost_map, &min, &max);
    mask = cost_map > (threshold*max) ;
    cost_map.copyTo(masked, mask);
    //cout << "Threshold value: " << threshold*max << endl;
}

Reconstruction::Reconstruction(Mat& map_x, Mat& map_y, Mat& right_image, Mat& left_image, stereoParams& stereo_params)
{
    // Initialize private members:
    stereoParams_ = stereo_params;
    leftImage_ = left_image;
    rightImage = right_image;
    mapX_ = map_x;
    mapY_ = map_y;
    warped_image_.create(rightImage.size(), rightImage.type());
    remap(right_image, warped_image_, mapX_, mapY_, CV_INTER_LINEAR, BORDER_CONSTANT, Scalar(0,0,0));
    
    // Create derivative maps:
    derivative(leftImage_, left_x_derivative_, "x");
    derivative(leftImage_, left_y_derivative_, "y");
    derivative(warped_image_, warped_x_derivative_, "x");
    derivative(warped_image_, warped_y_derivative_, "y");
    
    // Calculate correlstions:
    corr(leftImage_, warped_image_, 8, corr_intensities_);
    corr(left_x_derivative_, warped_x_derivative_, 8, corr_x_derivative_);
    corr(left_y_derivative_, warped_y_derivative_, 8, corr_y_derivative_);
    
    // Calculate covariance:
    variance(corr_intensities_, corr_x_derivative_, corr_y_derivative_, std_map_);
    
    // Calculate cost map:
    cost_function(corr_intensities_, corr_x_derivative_, corr_y_derivative_, std_map_, cost_map_);
    
    // Thresholding:
    threshold_filter(cost_map_, 0.4, masked_, mask_);
}

void Reconstruction::XYcalculate()
{
    Mat uv_temp, channel_u, channel_v;
    
    double focal_length_u    = stereoParams_.Kref.at<double>(0,0);
    double focal_length_v    = stereoParams_.Kref.at<double>(1,1);
    double principle_point_u = stereoParams_.Kref.at<double>(0,2);
    double principle_point_v = stereoParams_.Kref.at<double>(1,2);
    double skew              = stereoParams_.Kref.at<double>(0,1);
    double distance          = stereoParams_.d;
    
    findNonZero(mask_, uv_temp);
    uv_temp.convertTo(uv_temp, CV_64FC2);
    
    vector<Mat> channels(2);
    split(uv_temp, channels);
    channel_u = channels[0];
    channel_v = channels[1];
    
    channel_u = channel_u - principle_point_u;
    channel_v = channel_v - principle_point_v;
    
    hconcat(channel_u, channel_v, uv_);
    
    uv_ = uv_ * distance;
    
    double A_data[2][2] = {{focal_length_u, skew}, {0, focal_length_v}};
    Mat A(2,2,CV_64FC1,A_data);
    A = A.inv();
    transpose(A, A);
    
    //    cout << "channel_u: " << channel_u.rows << " , " << channel_u.cols << endl;
    //    cout << "channel_v: " << channel_v.rows << " , " << channel_v.cols << endl;
    //    cout << "A: " << A.rows << " , " << A.cols << endl;
    //    cout << "uv_ " << uv_.rows << " , " << uv_.cols << endl;
    
    XY_ = uv_*A;
    
    interpulated_Y_ = Y_interpulate();
}

double Reconstruction::Y_interpulate()
{
    // Calculate average of each row:
    Mat cols_average;
    reduce(XY_, cols_average, 0, CV_REDUCE_AVG);
    return cols_average.at<double>(0,1);
}

void Reconstruction::getWarped(Mat &dst)
{
    dst = warped_image_;
}

void Reconstruction::getCorrelationIntensities(Mat& dst)
{
    dst = corr_intensities_;
}

void Reconstruction::getCorrelation_X_Derivative(Mat& dst)
{
    dst = corr_x_derivative_;
}

void Reconstruction::getCorrelation_Y_Derivative(Mat& dst)
{
    dst = corr_y_derivative_;
}

void Reconstruction::getStdMap(Mat& dst)
{
    dst = std_map_;
}

void Reconstruction::getCostMap(Mat& dst)
{
    dst = cost_map_;
}

void Reconstruction::getMasked(Mat& dst)
{
    dst = masked_;
}

void Reconstruction::getMask(Mat& dst)
{
    dst = mask_;
}


double Reconstruction::getInterpulatedY()
{
    return interpulated_Y_;
}
