
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <time.h>
#include <omp.h>
#include"../../highPerTimer/HighPerfTimer.h"

typedef unsigned int ArrType_t;
using namespace std;

bool initializeCpu(ArrType_t**a, ArrType_t**b, ArrType_t**c, int arrSize); // this is where we initilize the three arrays on the cpu's memory
void cleanCpu(ArrType_t*a, ArrType_t*b, ArrType_t*c); // this is where we delete the arrays on the cpu's memory
void cleanGpu(ArrType_t*a, ArrType_t*b, ArrType_t*c);//free the gpu memory
void fillCPUArr(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize);
void addCPUVector(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize);
cudaError_t cudaSetMalloc(ArrType_t**a, ArrType_t**b, ArrType_t**c, unsigned int arrSize, cudaDeviceProp * devProps);
int main(int argc, char* argv[]){
	//seed random number generator for unique numbers
	srand(time(NULL));


	int size = 1000; // this is the default array size, if the user does not enter their own
	int rep = 100;// how many times to loop 
	
	cout << argc << endl;
	cout << argv[0] << endl;
	//create 3 int pointers on cpu and 3 on gpu
	ArrType_t *cpu_a = nullptr;
	ArrType_t *cpu_b = nullptr;
	ArrType_t *cpu_c = nullptr;

	ArrType_t *gpu_a = nullptr;
	ArrType_t *gpu_b = nullptr;
	ArrType_t *gpu_c = nullptr;

	cudaError_t gpuStatus = cudaSuccess;
	cudaDeviceProp gpuProps;

	if (argc > 1) {// check to see if the user entered a size of their own array
		size = stoi(argv[1]); // change default size to user specified size
		//cout << stoi(argv[1]) <<endl; //??IS NEEDED??check to make sure the size was changed 
	}
	cout << "array is of size " << size << endl; //check to make sure the size was changed,=

	
	//TRY
	try{
		//CPU TEST
		//allocate memory for the pointers of the user specified size
		bool init = initializeCpu(&cpu_a, &cpu_b, &cpu_c, size); //a bool is set to this so we know if the outcome worked or not
		if (init) {
			cout << "looking good" << endl; // memory was allocated correctly
		}
		else { // if on 
			throw("Initilaizing failed");//send this error message to catch block
		}
		
		//initialize timer, and the sum of the results
		double cpuRunSum = 0.0; // keeping a running timer of a sum for the cpu, so averaging will be super easy
		HighPrecisionTime hpt;//instance of timer
		hpt.TimeSinceLastCall();//call it for the first time to start it

		for (int i = 0; i < 100; i++) {//loop through and 
			//fill values in the pointers and 
			fillCPUArr(cpu_a, cpu_b, cpu_c, size);
			addCPUVector(cpu_a, cpu_b, cpu_c, size);
			cpuRunSum += hpt.TimeSinceLastCall();
		}
		//average the sum of each cpu run
		double cpuRunAvg = cpuRunSum / 100;
		cout << "Average time for filling arrays a and b then adding into c is " << cpuRunAvg << " ticks per second" << endl;


		//GPU TEST

				
											 
		//set dev and malloc arrs
		gpuStatus = cudaSetMalloc(&gpu_a, &gpu_b, &gpu_c,size, &gpuProps);//set it up

		//start timer for copy time from CPU to GPU
		
		//copy arrays and end timer

		//start timer and runing sum for gpu test timing 

		//loop through and add rep times

		//stop timer, find average
		
		//start timer for copy c time
		
		//copy c back to cpu and endtimer

	}//CATCH
	catch(char * errMessage){ 
		cout << "Error: " << errMessage <<endl;
		cleanCpu(cpu_a, cpu_b, cpu_c);
		cleanGpu(gpu_a, gpu_b, gpu_c);
	}
	




	//clean CPU at end, I assume this will move to an earlier part of the program when we get to cuda 
	cleanCpu(cpu_a, cpu_b, cpu_c);
	cleanGpu(gpu_a, gpu_b, gpu_c);
	system("pause");
	return 0;

}

bool initializeCpu(ArrType_t**a, ArrType_t**b, ArrType_t**c, int arrSize) {
	bool retval = true;

	int sizeType = sizeof(ArrType_t);
	*a = (ArrType_t*)malloc(arrSize * sizeType);
	*b = (ArrType_t*)malloc(arrSize * sizeType);
	*c = (ArrType_t*)malloc(arrSize * sizeType);

	if (*a == nullptr || *b == nullptr || *c == nullptr) {

		retval = false;
	}
			
	return retval;
}
void cleanCpu(ArrType_t* a, ArrType_t* b, ArrType_t*c) {
	
	if (a != nullptr) {
		free(a);
		a = nullptr;
		cout << "cpu_a was freed form CPU" << endl;
	}

	if (b != nullptr) {
		free(b);
		b = nullptr;
		cout << "cpu_b was freed from CPU" << endl;
	}

	if (c != nullptr) {
		free(c);
		c = nullptr;
		cout << "cpu_c was freed from CPU" << endl;
	}

}
void cleanGpu(ArrType_t* a, ArrType_t* b, ArrType_t*c) {

	cudaFree(&a);
	a = nullptr;
	cout << "gpu_a was freed form GPU?" << endl;
	
	cudaFree(&b);
	b = nullptr;
	cout << "gpu_b was freed from GPU?" << endl;

	cudaFree(&c);
	c = nullptr;
	cout << "gpu_c was freed from GPU?" << endl;
}
void fillCPUArr(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize) {

//#pragma omp parallel for

	for (int i = 0; i < arrSize; i++) { // loop through the amount of times as size, so to fill all the array slots
		a[i] = rand() % 10 +1;
		b[i] = rand() % 20 +1;
		c[i] = 0;
	}

}
void addCPUVector(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize) {

	for (int i = 0; i < arrSize; i++) {
		c[i] = a[i] + b[i];
	}

}
cudaError_t cudaSetMalloc(ArrType_t**gpu_a, ArrType_t**gpu_b, ArrType_t**gpu_c, unsigned int arrSize, cudaDeviceProp * devProp) {

	//initialize variables

	int mallocSize = arrSize * sizeof(ArrType_t);// make malloc easier to read, and to type
	cudaError_t cudaStatus; //holds 'cudaSuccess' or other to tell wether the cuda call worked or not.
	try {
		//set cuda device
		cudaStatus = cudaSetDevice(0); //use 0 bc we only have one graphics card 
		if (cudaStatus != cudaSuccess) // if the device did not set
		{//print error, and free gpu ptrs
			cout << "cudaSetDevice has failed" << endl;
			throw(1);
		}
		cudaStatus = cudaGetDeviceProperties(devProp, 0); //??? get props from gpu 0
		if (cudaStatus != cudaSuccess) {
			cout << "cudaGetDeviceProperties failed" << endl;
		}
		cout <<endl<< "cudaGetDeviceProperties worked" << endl;
		//cuda malloc a, b, c
		//a
		cudaStatus = cudaMalloc((void**)&gpu_a, mallocSize);
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMalloc gpu_a failed" << endl;
			throw(1);
		}
		cout << "gpu_ a was allocated" << endl;
		//b
		cudaStatus = cudaMalloc((void**)&gpu_b, mallocSize);
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMalloc gpu_b failed" << endl;
			throw(1);
		}
		cout << "gpu_b was allocated" << endl;
		//c
		cudaStatus = cudaMalloc((void**)&gpu_c, mallocSize);
		if (cudaStatus != cudaSuccess) {
			cout << "cudaMalloc gpu_c failed" << endl;
			throw(1);
		}
		cout << "gpu_c was allocated" << endl << endl;
		//DIFFernt function for coppy and kernel call and stuff bc that's timed
		//this is the end of the function
	}
	catch (int errNum) {//free the memory bc something didn't work
		cudaFree(gpu_a);
		cout << "gpu_a was freed" << endl;
		cudaFree(gpu_b);
		cout << "gpu_b was freed" << endl;
		cudaFree(gpu_c);
		cout << "gpu_c was freed" << endl;
	}


return cudaStatus;
}