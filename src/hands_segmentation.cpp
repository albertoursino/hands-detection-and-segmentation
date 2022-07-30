/**
@author Alberto Ursino

@date 29/07/2022
*/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <math.h>

using namespace cv;
using namespace std;

vector<Point> getWhitePixels(Mat binary_image) {
	vector<Point> white_pixels_coor;
	for (int i = 0; i < binary_image.rows; i++) {
		for (int j = 0; j < binary_image.cols; j++) {
			if (binary_image.at<unsigned char>(i, j) == 255)
				white_pixels_coor.push_back(Point(j, i));
		}
	}
	return white_pixels_coor;
}

double medianMat(Mat image) {
	image = image.reshape(0, 1);
	vector<double> vecFromMat;
	image.copyTo(vecFromMat);
	nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
	return vecFromMat[vecFromMat.size() / 2];
}

/*
This function applies the Canny's edge detector to an image after blurring it.

@param image rgb image
@return vector of edge points
*/
vector<Point> getEdges(Mat image) {
	Mat src_gray, edges, img_smoothed;

	bilateralFilter(image, img_smoothed, 9, 200, 200, BORDER_DEFAULT);
	cvtColor(img_smoothed, src_gray, COLOR_BGR2GRAY);

	// Finding best thresholds for the Canny's edge detector
	double v = medianMat(src_gray);
	double sigma = 0.33;
	int lower = static_cast<int>((1.0 - sigma) * v);
	lower = max(0, lower);
	int upper = static_cast<int>((1.0 + sigma) * v);
	upper = min(255, upper);

	Canny(img_smoothed, edges, lower, upper, 3, false);

	vector<Point> edge_points = getWhitePixels(edges);

	return edge_points;
}

vector<Point> getSeedPoints(Mat image) {
	Mat img = image.clone();
	Mat img_gray, img_otsu, img_eroded;
	vector<Point> seed_points;
	int erosion_size = 2;

	// Taking a smaller bounding box for a better seed points searching
	// since in the center of the image it's more likely to find hand pixels
	int crop_size_row = int(img.rows * 20 / 100);
	int crop_size_col = int(img.cols * 20 / 100);
	img = img(Rect(crop_size_col, crop_size_row,
		img.cols - 2 * crop_size_col, img.rows - 2 * crop_size_row));

	// Otsu's segmentation
	cvtColor(img, img_gray, COLOR_BGR2GRAY);
	threshold(img_gray, img_otsu, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// Removing boundaries pixels for a better erosion
	for (int i = 0; i < img_otsu.cols; i++) {
		img_otsu.at<unsigned char>(0, i) = 0;
		img_otsu.at<unsigned char>(img_otsu.rows - 1, i) = 0;
	}
	for (int i = 0; i < img_otsu.rows; i++) {
		img_otsu.at<unsigned char>(i, 0) = 0;
		img_otsu.at<unsigned char>(i, img_otsu.cols - 1) = 0;
	}

	// Eroding until there are less than 100 pixels
	int white_pixels = 101;
	Mat element = getStructuringElement(MORPH_RECT, Size(erosion_size, erosion_size));
	erode(img_otsu, img_eroded, element);
	while (white_pixels > 100) {
		white_pixels = 0;
		seed_points.clear();
		erode(img_eroded, img_eroded, element);
		for (int i = 0; i < img_eroded.rows; i++) {
			for (int j = 0; j < img_eroded.cols; j++) {
				if (img_eroded.at<unsigned char>(i, j) == 255) {
					seed_points.push_back(Point(j, i));
					white_pixels += 1;
				}
			}
		}
	}

	// Restoring original pixels coordinates
	for (int i = 0; i < seed_points.size(); i++) {
		seed_points.at(i).x += crop_size_col;
		seed_points.at(i).y += crop_size_row;
	}

	return seed_points;
}

/*
This function computes the mean color between a pixel and its neighbours.

@param image rgb image
@param pixel cv::Point
@param ksize kernel size
@return the mean color
*/
vector<float> getMeanColor(Mat image, Point pixel, int ksize = 2) {
	Mat img = image.clone();
	vector<Mat> channels(3);
	split(img, channels);

	float mean_r = 0;
	float mean_g = 0;
	float mean_b = 0;
	int count = 0;

	for (int i = -ksize; i <= ksize && (pixel.x + i < image.cols) && (pixel.x + i >= 0); i++) {
		for (int j = -ksize; j <= ksize && (pixel.y + j < image.rows) && (pixel.y + j >= 0); j++) {
			count += 1;
			mean_r += channels[0].at<unsigned char>(Point(pixel.x + i, pixel.y + j));
			mean_g += channels[1].at<unsigned char>(Point(pixel.x + i, pixel.y + j));
			mean_b += channels[2].at<unsigned char>(Point(pixel.x + i, pixel.y + j));
		}
	}

	vector<float> means = { mean_r / count, mean_g / count, mean_b / count };
	return means;
}

/**
This function computes the difference between the channels intensities of two pixels
and then sum them up. If two pixels have the same color the returned value is 0.

@param image rgb image
@param pixel1 first pixel coordinates
@param pixel2 second pixel coordinates
*/
int getColorDiff(Mat image, Point pixel1, Point pixel2) {
	Mat img = image.clone();
	vector<Mat> channels(3);
	split(img, channels);

	int x1 = abs(channels[0].at<unsigned char>(pixel1) - channels[0].at<unsigned char>(pixel2));
	int x2 = abs(channels[1].at<unsigned char>(pixel1) - channels[1].at<unsigned char>(pixel2));
	int x3 = abs(channels[2].at<unsigned char>(pixel1) - channels[2].at<unsigned char>(pixel2));

	return x1 + x2 + x3;
}

/**
This function states whether two pixels are similar or not.

@param image rgb image
@param pixel1 first pixel coordinates
@param pixel2 second pixel coordinates
@param th_mean threshold on the mean color difference
@param th_colordiff threshold on the color difference (see #get_color_diff)
*/
bool comparePixels(Mat image, Point pixel1, Point pixel2, int th_mean = 60, int th_colordiff = 70) {
	Mat img = image.clone();

	vector<float> mean1 = getMeanColor(img, pixel1);
	vector<float> mean2 = getMeanColor(img, pixel2);

	if (abs(mean1.at(0) - mean2.at(0)) < th_mean &&
		abs(mean1.at(1) - mean2.at(1)) < th_mean &&
		abs(mean1.at(2) - mean2.at(2)) < th_mean &&
		getColorDiff(image, pixel1, pixel2) < th_colordiff)
		return true;
	else
		return false;
}

/*
This function applies the region growing algorithm to an image.

@param image rgb image
@param seed_points cv::Point
@param edge_points cv::Point
@return binary image representing the region(s) grown from the seed points
*/
Mat regionGrowing(Mat image, vector<Point> seed_points, vector<Point> edge_points) {
	vector<Point> unexplored_points;
	vector<Point> explored_points;
	Mat result = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

	for (int s = 0; s < seed_points.size(); s++) {
		Point seed_point = seed_points.at(s);
		unexplored_points.push_back(seed_point);
		result.at<unsigned char>(seed_point) = 255;
		while (unexplored_points.size() > 0) {
			Point point = unexplored_points.at(unexplored_points.size() - 1);
			unexplored_points.pop_back();
			explored_points.push_back(point);
			for (int i = -1; i <= 1 && (point.x + i < image.cols) && (point.x + i >= 0); i++) {
				for (int j = -1; j <= 1 && (point.y + j < image.rows) && (point.y + j >= 0); j++) {
					Point neighbor = Point(point.x + i, point.y + j);
					if (neighbor != point) {
						if (!count(edge_points.begin(), edge_points.end(), neighbor)) {
							if (comparePixels(image, neighbor, seed_point)) {
								result.at<unsigned char>(neighbor) = 255;
								if (!count(explored_points.begin(), explored_points.end(), neighbor) &&
									!count(unexplored_points.begin(), unexplored_points.end(), neighbor))
									unexplored_points.push_back(neighbor);
							}
						}
					}
				}
			}
		}
	}
	return result;
}

/*
Component 1 (edge detector):
1. Bilateral filter + Canny's edge detector
Component 2 (region growing):
1. If the image is very small or quite large then performs a simple otsu's segmentation, else go to point 2
2. Seed point extraction
3. Region growing
4. Dilation

Region growing should not label as white pixels:
- Very different pixels from the starting seed (see "compare_pixels" method);
- Pixels on the edge lines found by component 1.

@param image rgb image of a detected hand
@return binary image representing the segmented hand
*/
Mat segmentHand(Mat image) {
	// Component 1
	vector<Point> detected_edges = getEdges(image);

	// Component 2
	Mat image_gray, hand_seg;
	int image_size = image.rows * image.cols;
	if (image_size < 2000 || image_size > 100000) {
		cvtColor(image, image_gray, COLOR_BGR2GRAY);
		threshold(image_gray, hand_seg, 0, 255, THRESH_BINARY | THRESH_OTSU);
	}
	else {
		vector<Point> seed_points = getSeedPoints(image);
		hand_seg = regionGrowing(image, seed_points, detected_edges);
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(2, 2));
		dilate(hand_seg, hand_seg, element);
	}

	return hand_seg;
}

Mat drawSegHands(Mat image, vector<Rect> b_boxes, Mat cropped_hand_seg, int index) {
	Mat img = image.clone();
	vector<Mat> channels(3);
	split(img, channels);

	// Defining a color palette
	vector<int> teal = { 123, 124, 14 };
	vector<int> tiffany = { 7, 207, 218 };
	vector<int> aero = { 84, 178, 46 };
	vector<int> carmine = { 70, 34, 214 };
	vector<int> dark_purple = { 98, 46, 118 };
	vector<vector<int>> colors = { teal, tiffany, aero, carmine, dark_purple };

	Rect box = b_boxes[index];
	int left = box.x;
	int top = box.y;

	for (int i = 0; i < cropped_hand_seg.rows; i++) {
		for (int j = 0; j < cropped_hand_seg.cols; j++) {
			if (cropped_hand_seg.at<unsigned char>(i, j) == 255) {
				channels[0].at<unsigned char>(top + i, left + j) =
					channels[0].at<unsigned char>(top + i, left + j) * 0.5 + colors[index % 5][0] * 0.5;
				channels[1].at<unsigned char>(top + i, left + j) =
					channels[1].at<unsigned char>(top + i, left + j) * 0.5 + colors[index % 5][1] * 0.5;
				channels[2].at<unsigned char>(top + i, left + j) =
					channels[2].at<unsigned char>(top + i, left + j) * 0.5 + colors[index % 5][2] * 0.5;
			}
		}
	}

	merge(channels, img);

	return img;
}