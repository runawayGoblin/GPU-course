
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <time.h>

typedef unsigned int ArrType_t;
using namespace std;

bool initialize(ArrType_t**a, ArrType_t**b, ArrType_t**c, int arrSize); // this is where we initilize the three arrays on the cpu's memory
void clean(ArrType_t*a, ArrType_t*b, ArrType_t*c); // this is where we delete the arrays on the cpu's memory
void fillArr(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize);

int main(int argc, char* argv[]){
	//seed random number generator for unique numbers
	srand(time(NULL));


	int size = 1000; // this is the default array size, if the user does not enter their own
	cout << argc << endl;
	cout << argv[0] << endl;

	if (argc > 1) {// check to see if the user entered a size of their own array

		size = stoi(argv[1]); // change default size to user specified size
		//cout << stoi(argv[1]) <<endl; //??IS NEEDED??check to make sure the size was changed 
	}
	cout << "array is of size " << size << endl; //check to make sure the size was changed,=

	//create 3 int pointers
	ArrType_t *a = nullptr;
	ArrType_t *b = nullptr;
	ArrType_t *c = nullptr;


	//TRY
	try{

		//allocate memory for the pointers of the user specified size
		bool init = initialize(&a, &b, &c, size); //a bool is set to this so we know if the outcome worked or not
		if (init) {
			cout << "looking good" << endl; // memory was allocated correctly
		}
		else { // if on 
			throw("Initilaizing failed");//send this error message to catch block
		}

		//fill values in the pointers
		fillArr(a, b, c, size);

		for (int j = 0; j < size; j++) {
			cout << "a[" << j << "] = " << a[j] << endl;
			cout << "b[" << j << "] = " << b[j] << endl;
			cout << "c[" << j << "] = " << c[j] << endl;
		}
		


	}//CATCH
	catch(char * errMessage){ 
		cout << "Error: " << errMessage;
		clean(a, b, c);
	}
	




	//clean CPU at end, I assume this will move to an earlier part of the program when we get to cuda 
	clean(a, b, c);

	system("pause");
	return 0;

}

bool initialize(ArrType_t**a, ArrType_t**b, ArrType_t**c, int arrSize) {
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
void clean(ArrType_t* a, ArrType_t* b, ArrType_t*c) {
	
	if (a != nullptr) {
		free(a);
		a = nullptr;
		cout << "a was freed form CPU" << endl;
	}

	if (b != nullptr) {
		free(b);
		b = nullptr;
		cout << "b was freed from CPU" << endl;
	}

	if (c != nullptr) {
		free(c);
		c = nullptr;
		cout << "c was freed from CPU" << endl;
	}

}
void fillArr(ArrType_t*a, ArrType_t*b, ArrType_t*c, int arrSize) {

	for (int i = 0; i < arrSize; i++) { // loop through the amount of times as size, so to fill all the array slots
		a[i] = rand() % 10 +1;
		b[i] = rand() % 20 +1;
		c[i] = 0;
	}

}
