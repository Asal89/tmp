#ifndef SRC_HOMOGRAPHY_H_
#define SRC_HOMOGRAPHY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utils.hpp"
using namespace std;
using namespace cv;



class homography {
    
public:
    
    homography(stereoParams& par);
    
    /*
     * Constructor TODO: add description
     * */
    
    const Mat& getHomography() const;
    /*
     * TODO: add description
     * */
    
private:
    
    Mat homogMat_;
};

#endif /* SRC_HOMOGRAPHY_H_ */
