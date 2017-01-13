#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>


using namespace cv;
using namespace std;

void CpuThreshold(int threshold, int width, int height, unsigned char* data);
cudaError_t mallocGPU(unsigned char** orig, unsigned char** modif, int width, int height);
void cleanGPU(unsigned char** orig, unsigned char**);
cudaError_t copyToGPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height);


int main(int argc, char** argv) {
	
	if (argc != 2) {
		cout << "usage: display_image ImageToLoadAndDisplay" << endl;
	}

	Mat cpuOriginalImage;
	cpuOriginalImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cvtColor(cpuOriginalImage, cpuOriginalImage, COLOR_RGB2GRAY);

	cout << "Number of channels: " << cpuOriginalImage.channels() << endl;
	if (!cpuOriginalImage.data) {
		cout << "could not open or find the image" << std::endl;
	}
	int height = cpuOriginalImage.rows;
	int width = cpuOriginalImage.cols;
	int threshold = 128;

	//declare vairables to put the data into
	//Mat.data returns a pointer to an unsighned character array
	unsigned char * gpuOriginalImage=0;
	unsigned char * gpuModifiedImage=0;
		
	CpuThreshold(threshold, width, height, cpuOriginalImage.data);
	cout << "CpuThreshold was a success" << endl << endl;
	try {
		cudaError_t gpuStatus = mallocGPU(&gpuOriginalImage, &gpuModifiedImage, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Malloc GPU Failed");
		}
		cout << "CpuThreshold was a success" << endl << endl;

		gpuStatus = copyToGPU(gpuOriginalImage, cpuOriginalImage.data, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Copy to GPU Failed");
		}
	}
	catch (char* errMsg) {
		cout << "Error: " << errMsg << endl;
		cleanGPU(&gpuOriginalImage, &gpuModifiedImage);
	}

	//namedWindow("Display Window", WINDOW_NORMAL);
	//imshow("Display Window", cpuOriginalImage);


	waitKey(0);
	return 0;
}
void CpuThreshold(int threshold, int width, int height, unsigned char* data) {

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


cudaError_t mallocGPU(unsigned char** orig, unsigned char** modif, int width, int height) {
	
	//variables for the malloccing 
	cudaError_t gpuStatus = cudaSuccess;
	int mallocSize = width * height * sizeof(unsigned char);
	cout << "malloc Size: " << mallocSize << endl;
	//set device and malloc data space
	try{
		gpuStatus = cudaSetDevice(0);
		if (gpuStatus != cudaSuccess) {
			throw("cudaSetDevice failed");
		}
		cout << "cudaSetDevice was a success" << endl << endl;
		//malloc space for originalImage.data
		gpuStatus = cudaMalloc((void**)&orig, mallocSize);
		if (gpuStatus != cudaSuccess) {
			throw("cudaMalloc gpuOriginal failed");
		}
		cout << "cudaMalloc gpuOriginal was a success" << endl << endl;
		//malloc space for modifiedImage.data
		gpuStatus = cudaMalloc((void**)&modif, mallocSize);
		if (gpuStatus != cudaSuccess) {
			throw("cudaMalloc gpuModified failed");
		}
		cout << "cudaMalloc gpuModified was a success" << endl << endl;
	}
	catch (char* errMsg) {
		cout << "Error: " << errMsg << endl << endl;
		cleanGPU(orig, modif);
	}

	return gpuStatus;
}
cudaError_t copyToGPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height) {
	cudaError_t copySucess = cudaSuccess;
	int cpySize = width * height * sizeof(unsigned char);
	cout << "Copy Size: "<<cpySize << endl;

	copySucess = cudaMemcpy(gpuData, cpuData, cpySize, cudaMemcpyHostToDevice);
	if (copySucess != cudaSuccess) {
		cout << "Error: cudaMemcpy failed" << endl << endl;
	}
	return copySucess;
}
void cleanGPU(unsigned char** orig, unsigned char** modif) {
	
	cudaError_t cleanStatus;
	
	cleanStatus = cudaFree(&orig);
	if (cleanStatus == cudaSuccess) {
		cout << "Origial Data was freed from the GPU" << endl;
	}
	cleanStatus = cudaFree(modif);
	if (cleanStatus == cudaSuccess) {
		cout << "Modified Data was freed from the GPU" << endl;
	}
}