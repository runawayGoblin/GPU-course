#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include"HighPerfTimer.h"




using namespace cv;
using namespace std;

typedef unsigned char UBYTE;

void CpuThreshold(int threshold, int width, int height, unsigned char* data);
cudaError_t mallocGPU(unsigned char** orig, unsigned char** modif, int width, int height);
void cleanGPU(unsigned char* orig, unsigned char*);
cudaError_t copyToGPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height);
cudaError_t copyToCPU(unsigned char* gpuData, unsigned char* cpuData, int width, int height);
void onTrack(int thr, void* pt);
void onTrackBar(int t, void* pt);
void BoxFilter(UBYTE* src, UBYTE* des, int width, int height, int* ker, int kw, int kh);

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
Mat cpuModifImage;
int height;
int width;
int kHeight;
int kWidth;
int thresholdSlider = 195;
HighPrecisionTime hpt;
int K3[] = { 1,1,1, 1,1,1, 1,1,1};
int K5[] = { 1,1,1,1,1, 1,1,1,1,1, 1,1,1,1,1,  1,1,1,1,1,   1,1,1,1,1, };
int K11[11*11] = {  };
int K19[19*19] = {  };
float boxSum = 0.0;
int boxTimes = 0;

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
	//CpuThreshold(thresholdSlider, width, height, cpuOriginalImage.data);
	//cout << "CpuThreshold was a success" << endl << endl;
	//try {
	//	//create space on the gpu to hold the modified and the image data
	//	cudaError_t gpuStatus = mallocGPU(&gpuOriginalImage, &gpuModifiedImage, width, height);
	//	if (gpuStatus != cudaSuccess) {
	//		throw("Malloc GPU Failed");
	//	}
	//	cout << "gpuMalloc was a success" << endl << endl;
	//	
	//	//copy the cpu image data to the gpu
	//	gpuStatus = copyToGPU(gpuOriginalImage, cpuOriginalImage.data, width, height);
	//	if (gpuStatus != cudaSuccess) {
	//		throw("Copy to GPU Failed");
	//	}
	//	cout << "cudaMemcpy cpuImg to gpuOrig Worked" << endl << endl;

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


	/*}
	catch (char* errMsg) {
		cout << "Error: " << errMsg << endl;
		cleanGPU(gpuOriginalImage, gpuModifiedImage);
	}
	*/ 

	cpuModifImage = cpuOriginalImage.clone();
	Mat temp = cpuOriginalImage.clone();

	for (int i = 0; i < (11 * 11); i++) {
		K11[i] = 1;
	}
	for (int i = 0; i < (19 * 19); i++) {
		K19[i] = 1;
	}

	//TESTING FLIPPED KERNEL
	/*int k2[] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	imshow("dispaly", cpuOriginalImage);
	waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K, kHeight, kHeight);
	imshow("dispaly", cpuModifImage);
	waitKey(0);
	BoxFilter(cpuOriginalImage.data, temp.data, width, height, k2, kHeight, kHeight);
	imshow("dispaly", temp);
	waitKey(0);
	temp= temp + cpuModifImage;
	imshow("dispaly", temp);
	waitKey(0);
*/
	namedWindow("Display Window", WINDOW_NORMAL);
	imshow("Display Window", cpuOriginalImage);
	waitKey(0);
	createTrackbar("Slider", "Display Window", &thresholdSlider, 255, onTrackBar);
	onTrackBar(thresholdSlider, 0);

	waitKey(0);
	cout << endl << "IMGAGE SIZE: " << width << " * " << height << endl;
	cout << "KERNEL SIZE: 9" << endl;
	cout << "AVGERAGE TIME: " << boxSum / boxTimes << endl;
	//cleanGPU(gpuOriginalImage, gpuModifiedImage);
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
	
	//cout << "threshold: " << thresholdSlider << endl;

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
void onTrackBar(int t, void* pt) {
	cout << "in track bar" << endl;
	//update num times box filter was ran
	boxTimes++;
	//start timer before box filter call
	hpt.TimeSinceLastCall();

	//run box filter
	//BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K3,3, 3);

	//waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K5, 5, 5);

	waitKey(0);
	
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K11, 11, 11);


	//waitKey(0);
	//BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K19, 19, 19);

	//find how long last iteration took
	float timeSinceLast = hpt.TimeSinceLastCall();
	cout << "Iteration " << boxTimes << ": " << timeSinceLast << "Second" << endl;
	//find the average box filter time
	boxSum += timeSinceLast;
	cout << "Average Time:" << boxSum / boxTimes << " seconds" << endl<<endl;

	imshow("Display Window", cpuModifImage);

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
	
	unsigned char* cpuOrig = cpuData;
	//copy data of the modified image from the gpu back to the gpu
	copySuccess = cudaMemcpy(cpuData, gpuData, copySize, cudaMemcpyDeviceToHost);
	if (copySuccess != cudaSuccess) {
		cout << "Error: cudaMemcpy Gpu to Cpu failed" << endl << endl;
	}
	cout << "a" << endl;
	if (cpuOrig != cpuData) {
		cout << "Copy to CPU Worked at Threshold " << thresholdSlider << endl;
	}
	cout << "b"<< endl;
	return copySuccess;
}

void BoxFilter(UBYTE* src, UBYTE* des, int width, int height, int* ker, int kw, int kh) {

	//this is bluring i thinhk, so yeah
	int runSum = 0;
	int kSum = 0;
	
	int pos = 0;
	int hEdge = kh / 2;
	int wEdge = kw / 2 ;
	// step through the entire image, every pixel
	for (int sH = hEdge; sH < height - hEdge; sH++) {
		for (int sW = wEdge; sW < width - wEdge; sW++) {
			//cout << width << "(" << sH << "," << sW << ")" << endl;
			//at each pixel grab the 8 surrounding pixel 
			//if the pixel is in the middle of the image and has no restrictions
				//top left
			pos = sH*width + sW;
			for (int i = 0; i < kw * kh; i++) {
				kSum += ker[i];//increment the denominator (???)WOULD seperatea loop outside (run once) be more efficient(???)
				//POS in kernel MOD kernel width - wedge (max number of spaces to move)	
				int x = ((i% kw) - wEdge);//how many times we move over
				//POS in kernel DIV kernal height - hedge(max num spaces to move up or down
				int y =((i/kw) - hEdge); 
					y = width * y;// will aways be negative bc y is neg
				runSum += src[pos + x +y] * ker[i];
				//top middle
			
			}
			//add all the surrounding values and then divide my num values to get concentration of blur
			//cout << sW *sH << "of " << width * height << endl;
			if (kSum != 0) {
				//cout << "A:(" << sH << "," << sW << "): " << runSum << "  :  " << kSum << "  :  "  <<int((float)runSum / (float)kSum) << endl;
				des[pos] = int((float)runSum / (float)kSum);
				runSum = 0;
			}
			else {
				//cout << "B:(" << sH << "," << sW << "): " << int((float)runSum / 1.0f) << endl;
				des[pos] = int((float)runSum / 1.0f);
				runSum = 0;
			}
			kSum = 0;//set kSum back to 0 or else the divisions would be too small, it would be 0
		}
	}




}