#ifndef HANDS_DETECTION
#define HANDS_DETECTION

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

vector<Mat> detectHands(Mat& input_image, Net& net);

vector<Rect> getBBoxes(Mat& input_image, vector<Mat>& outputs);

Mat drawBBoxes(Mat image, vector<Rect> b_boxes);

#endif