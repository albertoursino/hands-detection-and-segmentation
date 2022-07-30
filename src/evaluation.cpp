#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "hands_detection.h"
#include "hands_segmentation.h"
#include "utils.h"

using namespace cv;
using namespace std;
using namespace cv::dnn;


int main(int argc, char** argv) {
	//if (argc < 2) {
	//	cout << "Not enough command arguments" << endl;
	//	// Waits any key to be pressed
	//	cin.get();
	//	return -1;
	//}
	//String IMAGE_PATH = argv[1];

	vector<String> images, truth;
	glob("dataset/test_set/*.jpg", images, false);
	glob("dataset/seg_truth/*.png", truth, false);

	// Loading model and predicting bounding boxes
	Net net = readNet("best2.onnx");

	int count = images.size(); //number of png files in images folder

	std::ofstream myfile;
	myfile.open("segm_acc.csv");
	myfile << "accuracy\n";

	for (int j = 20; j < 30; j++) {
		cout << ">>>>> Image " << j << " of " << count << endl;
		Mat source = imread(images[j]);

		// ---------------Detection---------------
		//try {
		//	Mat source = imread(IMAGE_PATH);
		//} catch (int e) {
		//	cout << "An exception occurred. Exception Nr. " << e << '\n';
		//}
		

		//namedWindow("input image", WINDOW_NORMAL);
		//imshow("input image", source);

		vector<Mat> detections = detectHands(source, net);
		vector<Rect> b_boxes = getBBoxes(source, detections);

		Mat detected_hands = drawBBoxes(source, b_boxes);

		//namedWindow("human hands detection", WINDOW_NORMAL);
		//imshow("human hands detection", detected_hands);

		cout << "Detection completed; detected " << b_boxes.size() << " hands." << endl;
		// ---------------Segmentation---------------
		Mat segmented_hands = source.clone();
		Mat full_mask = Mat::zeros(Size(source.cols, source.rows), CV_8UC1);;
		for (int i = 0; i < b_boxes.size(); i++) {
			// Doing segmentation on each cropped hand
			Mat cropped_hand = source(b_boxes[i]);
			Mat cropped_hand_seg = segmentHand(cropped_hand);

			cout << "Segmentation " << i + 1 << " of " << b_boxes.size() << " Done." << endl;
			segmented_hands = drawSegHands(segmented_hands, b_boxes, cropped_hand_seg, i);

			// Preapring the full mask for evaluating the pixel accuracy
			Rect box = b_boxes[i];
			int left = box.x;
			int top = box.y;
			for (int k = 0; k < cropped_hand_seg.rows; k++) {
				for (int z = 0; z < cropped_hand_seg.cols; z++) {
					if (cropped_hand_seg.at<unsigned char>(k, z) == 255)
						full_mask.at<uchar>(top + k, left + z) = 255;
				}
			}

		}

		//namedWindow("human hands segmentation", WINDOW_NORMAL);
		//imshow("human hands segmentation", segmented_hands);

		// Setting side by side the two images
		Mat result = Mat(Size(2 * detected_hands.cols, detected_hands.rows), detected_hands.type());
		Mat left_img(result, Rect(0, 0, detected_hands.cols, detected_hands.rows));
		detected_hands.copyTo(left_img);
		Mat right_img(result, Rect(detected_hands.cols, 0, segmented_hands.cols, segmented_hands.rows));
		segmented_hands.copyTo(right_img);

		//imshow("Hand Detection + Recognition", result);
		imwrite("dataset/saved_images/" + to_string(j) + ".jpg", result);

		float acc = pixelAccuracy(full_mask, imread(truth[j], IMREAD_GRAYSCALE));

		myfile << to_string(acc) + "\n";
	}

	myfile.close();
	waitKey(0);
}