#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>


using namespace cv;
using namespace std;

void Threshold(int threshold, int width, int height, unsigned char* data);

int main(int argc, char** argv) {
	
	if (argc != 2) {
		cout << "usage: display_image ImageToLoadAndDisplay" << endl;
	}

	Mat image;
	image = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	cvtColor(image, image, COLOR_RGB2GRAY);

	cout << "Number of channels: " << image.channels() << endl;
	if (!image.data) {
		cout << "could not open or find the image" << std::endl;
	}
	int height = image.rows;
	int width = image.cols;
	int threshold = 128;

	Threshold(threshold, width, height, image.data);

	namedWindow("Display Window", WINDOW_NORMAL);
	imshow("Display Window", image);


	waitKey(0);
	return 0;
}
void Threshold(int threshold, int width, int height, unsigned char* data) {

	unsigned char* endArr = data + (width *height);
	//loop through data 
	for (unsigned char * index = data; index < endArr; index++) {
		//if data at index is greater than threshold index=255
		if (*index > threshold) {
			*index = 255;
		}
		else {//otherwise index = 0
			*index = 0;
		}
		
	}

}