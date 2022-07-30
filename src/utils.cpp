#include <opencv2/opencv.hpp>

using namespace cv;

/*
* (n° of hand pixels classified as hand + n° of non - hand pixels classified as non - hand) / (total n° of pixels)
*/
float pixelAccuracy(Mat prediction, Mat truth)
{
    float count = 0.;
    int width = prediction.cols;
    int height = prediction.rows;
    for (int j = 0; j < width; j++)
    {
        for (int k = 0; k < height; k++)
        {
            if (prediction.at<uchar>(k, j) == truth.at<uchar>(k, j))
                count += 1.;
        }
    }
    float pixel_accuracy = count / (width * height);
    return pixel_accuracy;
}