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
bool runGPUboxFilter();
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
__constant__ int gpuKernelHeight;
__constant__ int gpuKernelWidth;
__constant__ int gpuKernel[9];

__global__ void gpuBoxFilter(UBYTE* src, UBYTE* des, int width, int height) {
	//GET THE INDEX
	int index = blockIdx.x * blockDim.x + threadIdx.x;


	//CHECK if in bounds
	if (index < height && index < width) {
		int runSum = 0;
		int kSum = 0;

		int pos = 0;
		int hEdge = gpuKernelHeight / 2;
		int wEdge = gpuKernelWidth / 2;
		for (int i = 0; i < gpuKernelWidth * gpuKernelHeight; i++) {
			kSum += gpuKernel[i];//increment the denominator (???)WOULD seperatea loop outside (run once) be more efficient(???)
		}

		// step through the entire image, every pixel
		for (int sH = hEdge; sH < height - hEdge; sH++) {
			for (int sW = wEdge; sW < width - wEdge; sW++) {
				//at each pixel grab the kernelSize-1 surrounding pixel 
				//if the pixel is in the middle of the image and has no restrictions
				pos = sH*width + sW; //get the pos in the picter the pixel is at
				for (int i = 0; i < gpuKernelWidth * gpuKernelHeight; i++) {
					//POS in kernel MOD kernel width - wedge (max number of spaces to move)	
					int x = ((i% gpuKernelWidth) - wEdge);//how many times we move over
					//POS in kernel DIV kernal height - hedge(max num spaces to move up or down
					int y = ((i / gpuKernelWidth) - hEdge);//how many times we move up

					y = width * y;// will aways be negative if y is neg & always pos if y is pos
					runSum += src[pos + x + y] * gpuKernel[i];
					//top middle

				}
				//add all the surrounding values and then divide my num values to get concentration of blur
				if (kSum != 0) {
					des[pos] =int((float)runSum / (float)kSum);
					runSum = 0;
				}
				else {
					des[pos] = int((float)runSum / 1.0f);
					runSum = 0;
				}
				kSum = 0;//set kSum back to 0 or else the divisions would be too small, it would be 0, nothing would work
			}
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
float sumCPU = 0.0;
int timesCPU = 0;
float sumGPU = 0.0;
int timesGPU = 0;


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
	
	//TEST ALL the different variables(sizes) of kernel
/*	
	Mat temp = cpuOriginalImage.clone();
	for (int i = 0; i < (11 * 11); i++) {
		K11[i] = 1;
	}
	for (int i = 0; i < (19 * 19); i++) {
		K19[i] = 1;
	}
	*/
	cpuModifImage = cpuOriginalImage.clone();
	namedWindow("Display Window", WINDOW_NORMAL);
	imshow("Display Window", cpuModifImage);//show original image
	waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K3, 3, 3);
	imshow("Display Window", cpuModifImage);//show modified img
	waitKey(0);
	//set modified image back to original so you know there will be a change
	cpuModifImage = cpuOriginalImage.clone();

	/*
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K5, 5, 5);
	imshow("Display Window", cpuModifImage);
	waitKey(0);

	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K11, 11, 11);

	imshow("Display Window", cpuModifImage);
	waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K19, 19, 19);
*/

	
	imshow("Display Window", cpuModifImage);// will show the modified image
	waitKey(0);// show orig again
	//createTrackbar("Slider", "Display Window", &thresholdSlider, 255, onTrackBar);
	//onTrackBar(thresholdSlider, 0);
	//waitKey(0);

	if (runGPUboxFilter()) {
		imshow("Display Window", cpuModifImage);
		waitKey(0);
		cout << endl << "IMGAGE SIZE: " << width << " * " << height << endl;
		cout << "KERNEL SIZE: 9" << endl;
		cout << "AVGERAGE CPU TIME: " << sumCPU / timesCPU << endl;
		cout << "AVERAGE GPU TIME: " << sumGPU / timesGPU << endl <<endl;
	}
	else {
		cout << "GPU failed to run correctly" << endl;
	}
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
	timesCPU++;
	//start timer before box filter call
	hpt.TimeSinceLastCall();

	//run box filtertimesCPU
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K3,3, 3);

	waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K5, 5, 5);

	waitKey(0);
	
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K11, 11, 11);


	waitKey(0);
	BoxFilter(cpuOriginalImage.data, cpuModifImage.data, width, height, K19, 19, 19);

	//find how long last iteration took
	float timeSinceLast = hpt.TimeSinceLastCall();
	cout << "Iteration " << timesCPU << ": " << timeSinceLast << "Second" << endl;
	//find the average box filter time
	sumCPU += timeSinceLast;
	cout << "Average Time:" << sumCPU / timesCPU << " seconds" << endl<<endl;

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
	cout << "Copy To CPU Size: " << copySize << endl;
	
	unsigned char* cpuOrig = cpuData;
	//copy data of the modified image from the gpu back to the gpu
	copySuccess = cudaMemcpy(cpuData, gpuData, copySize, cudaMemcpyDeviceToHost);
	if (copySuccess != cudaSuccess) {
		cout << "~\n ~\nError: cudaMemcpy Gpu to Cpu failed~\n ~" << endl << endl;
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
bool runGPUboxFilter() {
	bool retval = true;

	try {
		cudaError_t gpuStatus = mallocGPU(&gpuOriginalImage, &gpuModifiedImage, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Malloc GPU Failed");
		}
		if (gpuOriginalImage == nullptr) {
			throw("gpuOriginalImage is null after malloc");
		}
		if (gpuModifiedImage == nullptr) {
			throw("gpuModifiedImage is null after malloc");
		}
		cout << "gpuMalloc was a success" << endl << endl;
			
		//copy the cpu image data to the gpu
		gpuStatus = copyToGPU(gpuOriginalImage, cpuOriginalImage.data, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Copy to GPU Failed");
		}
		cout << "cudaMemcpy cpuImg to gpuOrig Worked" << endl << endl;
		

		//send kernel and kernel size to the gpu constant memory
		if (cudaMemcpyToSymbol(gpuKernelHeight, &kHeight, sizeof(int)) != cudaSuccess) {
			throw("cudaMemcpyToSymbol kernelHeight failed");
		}
		cout << "cudaMemcpyToSymbol kernelHeight worked" << endl;

		if (cudaMemcpyToSymbol(gpuKernelWidth, &kWidth, sizeof(int)) != cudaSuccess) {
			throw("cudaMemcpyToSymbol kernelWidth failed");
		}
		cout << "cudaMemcpyToSymbol kernelWidth worked" << endl;
		if (cudaMemcpyToSymbol(gpuKernel, &K3, sizeof(int)*9) != cudaSuccess) {
			throw("cudaMemcpyToSymbol kernelHeight failed");
		}
		cout << "cudaMemcpyToSymbol K3 worked" << endl;



		////update the image with the threshold
		int numBlocks = (1023 + width * height) / 1024;
		gpuBoxFilter <<<numBlocks, 1024 >>> (gpuOriginalImage, gpuModifiedImage, width, height);
		cudaDeviceSynchronize();
		//time(??)
		gpuStatus = cudaGetLastError();
		if (gpuStatus != cudaSuccess) {
			throw("Kernel Failed");
		}
		//cout << "Kernel Worked" << endl << endl;


		//copy back to cpu
		gpuStatus = copyToCPU(gpuModifiedImage, cpuModifImage.data, width, height);
		if (gpuStatus != cudaSuccess) {
			throw("Copy to GPU Failed");
		}
		cout << "cudaMemcpu to Cpu Worked" << endl << endl;



	}
	catch (char* err) {
		cleanGPU(gpuOriginalImage, gpuModifiedImage);
		retval = false;
	}

	return retval;
}