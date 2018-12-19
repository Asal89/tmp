#include "homography.h"



homography::homography(stereoParams& par){
    //TODO: check correctness with matlab
    this->homogMat_ = par.K * (par.r - (1/par.d) * par.t * par.n) * par.Kref.inv();
}

const Mat& homography::getHomography() const{
    return homogMat_;
}
