#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;
using namespace cv::dnn;



vector<Mat> detectHands(Mat& input_image, Net& net)
{
	// Passing as input the image as 640x640 blob
	Mat blob;
	blobFromImage(input_image, blob, (1. / 255.), Size(640, 640), Scalar(), true, false);
	net.setInput(blob);
	vector<Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());
	return outputs;
}

/*
Finds all bounding boxes inside the prediction and selects the relevant ones.
*/
vector<Rect> getBBoxes(Mat& input_image, vector<Mat>& outputs)
{

	float SCORE_THRESHOLD = 0.5;
	float CONFIDENCE_THRESHOLD = 0.5;

	vector<float> confidences;
	vector<Rect> boxes;
	float horiz_scale = input_image.cols / 640.0;
	float vert_scale = input_image.rows / 640.0;
	float* data = (float*)outputs[0].data;

	int elements = 25200; // it's the output dimension for 640x640 input (25200 bounding boxes)
	for (int i = 0; i < elements; i++)
	{
		// Collecting outputs values
		float center_x = data[0];
		float center_y = data[1];
		float width_norm = data[2];
		float height_norm = data[3];
		float confidence = data[4];
		float score = data[5];
		
		if (confidence >= CONFIDENCE_THRESHOLD)
		{
			
			if (score > SCORE_THRESHOLD)
			{
				// Storing boxes with score over defined threshold
				int topleft_x = static_cast<int>((center_x - 0.5 * width_norm) * horiz_scale);
				int topleft_y = static_cast<int>((center_y - 0.5 * height_norm) * vert_scale);
				int width = static_cast<int>(width_norm * horiz_scale);
				int height = static_cast<int>(height_norm * vert_scale);

				boxes.push_back(Rect(topleft_x, topleft_y, width, height));
				confidences.push_back(confidence);
			}
		}
		data += 6; //Iterating to the next row of outputs
	}
	// Filtering out overlapping boxes, keeping the best ones
	float NMS_THRESHOLD = 0.45;
	vector<int> indices;
	NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

	vector<Rect> bboxes;
	for (int i = 0; i < indices.size(); i++)
		bboxes.push_back(boxes[indices[i]]);

	return bboxes;
}


Mat drawBBoxes(Mat image, vector<Rect> b_boxes) {
	Mat to_draw = image.clone();
	for (int i = 0; i < b_boxes.size(); i++) {
		Rect box = b_boxes[i];
		int topleft_x = box.x;
		int topleft_y = box.y;
		int width = box.width;
		int height = box.height;
		// Draw bounding box.
		rectangle(to_draw, Point(topleft_x, topleft_y), Point(topleft_x + width, topleft_y + height), Scalar(255,0,0), 2);
	}
	return to_draw;
}