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
void cleanGPU(unsigned char* orig, unsigned char*);
cudaError_t copyToGPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height);
cudaError_t copyToCPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height);
void onTrack(int thr, void* pt);
__global__ void gpuThresholdKernel(unsigned char* gpuOrig, unsigned char* gpuModif, int sizeArr, int threshold) {
	//loop, compare threshold, change vals
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < sizeArr) {
		if (gpuOrig[i] > threshold) {
			gpuModif[i] = 255;
		}
		else {
			gpuModif[i] = 0;
		}
	}



}
//global vars
unsigned char * gpuOriginalImage = nullptr;
unsigned char * gpuModifiedImage = nullptr;
Mat cpuOriginalImage;
int height;
int width;
int thresholdSlider = 195;

int main(int argc, char** argv) {
	
	if (argc != 2) {
		cout << "usage: display_image ImageToLoadAndDisplay" << endl;
	}

	/*Mat cpuOriginalImage;*/
	cpuOriginalImage = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	cvtColor(cpuOriginalImage, cpuOriginalImage, COLOR_RGB2GRAY);

	cout << "Number of channels: " << cpuOriginalImage.channels() << endl;
	if (!cpuOriginalImage.data) {
		cout << "could not open or find the image" << std::endl;
	}
	height = cpuOriginalImage.rows;
	width = cpuOriginalImage.cols;
	//int threshold = 14;

	//declare vairables to put the data into
	//Mat.data returns a pointer to an unsighned character array
	/*unsigned char * gpuOriginalImage;
	unsigned char * gpuModifiedImage=0;
	*/	
	CpuThreshold(thresholdSlider, width, height, cpuOriginalImage.data);
	cout << "CpuThreshold was a success" << endl << endl;
	try {
		//create space on the gpu to hold the modified and the image data
		cudaError_t gpuStatus = mallocGPU(&gpuOriginalImage, &gpuModifiedImage, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Malloc GPU Failed");
		}
		cout << "gpuMalloc was a success" << endl << endl;
		
		//copy the cpu image data to the gpu
		gpuStatus = copyToGPU(gpuOriginalImage, cpuOriginalImage.data, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Copy to GPU Failed");
		}
		cout << "cudaMemcpy cpuImg to gpuOrig Worked" << endl << endl;

		//////update the image with the threshold
		//int numBlocks = (1023 + width * height) / 1024;
		//gpuThresholdKernel <<<numBlocks, 1024 >>> (gpuOriginalImage, gpuModifiedImage,(width * height), thresholdSlider);
		////time(??)
		//gpuStatus = cudaGetLastError();
		//if (gpuStatus != cudaSuccess) {
		//	throw("Kernel Failed");
		//}
		////cout << "Kernel Worked" << endl << endl;


		////copy back to cpu
		//gpuStatus = copyToCPU(gpuOriginalImage, cpuOriginalImage.data, width, height);
		//if (gpuStatus != cudaSuccess) {
		//	throw("Copy to GPU Failed");
		//}
		//cout << "cudaMemcpu to Cpu Worked" << endl << endl;


	}
	catch (char* errMsg) {
		cout << "Error: " << errMsg << endl;
		cleanGPU(gpuOriginalImage, gpuModifiedImage);
	}
	
	namedWindow("Display Window", WINDOW_NORMAL);
	//imshow("Display Window", cpuOriginalImage);
	createTrackbar("Slider", "Display Window", &thresholdSlider, 255, onTrack);
	onTrack(thresholdSlider, 0);

	waitKey(0);
	cleanGPU(gpuOriginalImage, gpuModifiedImage);
	return 0;
}
void CpuThreshold(int threshold, int width, int height, unsigned char* data) {

	unsigned char* endArr = data + (width * height);
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
void onTrack(int, void* ) {
	cudaError_t gpuStatus = cudaSuccess;
	
	int numBlocks = (1023 + width * height) / 1024;
	gpuThresholdKernel << <numBlocks, 1024 >> > (gpuOriginalImage, gpuModifiedImage, (width * height), thresholdSlider);
	
	gpuStatus = cudaGetLastError();
	if (gpuStatus != cudaSuccess) {
		cout << "nopeA" << endl;
	}
	
	cout << "threshold: " << thresholdSlider << endl;

	gpuStatus = cudaDeviceSynchronize();
	if (gpuStatus != cudaSuccess){
		cout << "ooooopppppss" << endl;
	}

	//copy back to cpu
	gpuStatus = copyToCPU(gpuModifiedImage, cpuOriginalImage.data, width, height);
	if (gpuStatus != cudaSuccess) {
		cout << "gpu copy failed" << endl;
		cleanGPU(gpuOriginalImage, gpuModifiedImage);
	}
	//cout << "cudaMemcpu to Cpu Worked" << endl << endl;
	imshow("Display Window", cpuOriginalImage);
}
cudaError_t mallocGPU(unsigned char** orig, unsigned char** modif, int width, int height) {
	
	//variables for the mallocing 
	cudaError_t gpuStatus = cudaSuccess; //test var
	int mallocSize = width * height * sizeof(unsigned char);//size var, for readablity
	cout << "malloc Size: " << mallocSize << endl;//test the size malloced
	//set device and malloc data space
	try{
		gpuStatus = cudaSetDevice(0);
		if (gpuStatus != cudaSuccess) {
			throw("cudaSetDevice failed");
		}
		cout << "cudaSetDevice was a success" << endl;

		//malloc space for originalImage.data
			//NOT & Bc that is passing the adress of the thing holding the adress of the pointer, and all else gets lost 
		gpuStatus = cudaMalloc((void**)orig, mallocSize); 
		if (gpuStatus != cudaSuccess) {
			throw("cudaMalloc gpuOriginal failed");
		}
		cout << "cudaMalloc gpuOriginal was a success" << endl;
		//malloc space for modifiedImage.data
		gpuStatus = cudaMalloc((void**)modif, mallocSize);
		if (gpuStatus != cudaSuccess) {
			throw("cudaMalloc gpuModified failed");
		}
		cout << "cudaMalloc gpuModified was a success" << endl;
	}
	catch (char* errMsg) {
		cout << "Error: " << errMsg << endl << endl;
		cleanGPU(*orig, *modif);
	}

	return gpuStatus;
}
cudaError_t copyToGPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height) {
	cudaError_t copySucess = cudaSuccess;//test var
	int cpySize = width * height * sizeof(unsigned char);//size var, for readability
	cout << "Copy to GPU Size: "<<cpySize << endl;//test size

	//copy the data from the original image onto the malloced space on the gpu
	copySucess = cudaMemcpy(gpuData, cpuData, cpySize, cudaMemcpyHostToDevice);
	if (copySucess != cudaSuccess) {
		cout << "Error: cudaMemcpy Cpu to Gpu failed" << endl << endl;
	}

	return copySucess;
}
void cleanGPU(unsigned char* orig, unsigned char* modif) {
	
	cudaError_t cleanStatus= cudaSuccess;
	
	cleanStatus = cudaFree(orig);
	if (cleanStatus == cudaSuccess) {
		cout << "Origial Data was freed from the GPU" << endl;
	}
	cleanStatus = cudaFree(modif);
	if (cleanStatus == cudaSuccess) {
		cout << "Modified Data was freed from the GPU" << endl;
	}
}
cudaError_t copyToCPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height) {
	cudaError_t copySuccess= cudaSuccess;
	int copySize = width * height * sizeof(unsigned char);
	//cout << "Copy To CPU Size: " << copySize << endl;
	
	//copy data of the modified image from the gpu back to the gpu
	copySuccess = cudaMemcpy(cpuData, gpuData, copySize, cudaMemcpyDeviceToHost);
	if (copySuccess != cudaSuccess) {
		cout << "Error: cudaMemcpy Gpu to Cpu failed" << endl << endl;
	}

	return copySuccess;
}