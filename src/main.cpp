/*
@author Alberto Ursino & Fabio Marangoni

@date 29/07/2022
*/

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "hands_detection.h"
#include "hands_segmentation.h"
#include "utils.h"

#define CV_VERSION

using namespace cv;
using namespace std;
using namespace cv::dnn;

int main(int argc, char** argv) {

	String MODEL_NAME = "trained_yolo.onnx";

	// READING INPUT IMAGE

	if (argc < 2) {
		cout << "Not enough command arguments" << endl;
		// Waits any key to be pressed
		cin.get();
		return -1;
	}
	String IMAGE_PATH = argv[1];
	Mat source;
	try {
		source = imread(IMAGE_PATH);
	}
	catch (int e) {
		cout << "An exception occurred. Exception Nr. " << e << '\n';
	}
	namedWindow("input image", WINDOW_AUTOSIZE);
	imshow("input image", source);

	// DETECTION

	// Loading model and predicting bounding boxes
	Net net = readNet(MODEL_NAME);
	vector<Mat> detections = detectHands(source, net);
	vector<Rect> b_boxes = getBBoxes(source, detections);

	Mat detected_hands = drawBBoxes(source, b_boxes);

	namedWindow("human hands detection", WINDOW_AUTOSIZE);
	imshow("human hands detection", detected_hands);

	// SEGMENTATION

	Mat segmented_hands = source.clone();
	Mat full_mask = Mat::zeros(Size(source.cols, source.rows), CV_8UC1);;
	for (int i = 0; i < b_boxes.size(); i++) {
		// Doing segmentation on each cropped hand
		Mat cropped_hand = source(b_boxes[i]);
		Mat cropped_hand_seg = segmentHand(cropped_hand);

		segmented_hands = drawSegHands(segmented_hands, b_boxes, cropped_hand_seg, i);
	}

	namedWindow("human hands segmentation", WINDOW_AUTOSIZE);
	imshow("human hands segmentation", segmented_hands);

	waitKey(0);
}