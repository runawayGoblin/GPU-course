
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "highPerfTimer.h"
#include <stdio.h>
#include <fstream>
#include <stdlib.h>
#include <iostream>
using namespace std;


int main() {
	//create the thing with the stuff for the position
	int giga = 1 << 30;


	//add timer
	HighPrecisionTime hpt;

	//array of char, this will be a buffer
	char *buffer = nullptr; // (char*)malloc(giga);
	buffer = new char[giga]();
	//create bitmap
	char * bitmap = nullptr;
	bitmap = new char[giga / 8]();


	//int * bitmap = (int *)malloc(giga / 8);

	//open file
	ifstream enWiki("C:/Users/educ/Documents/enwiki-latest-abstract.xml");
	if (enWiki.fail()) {
		cout << "Cannot open file " << endl;
		return 1;
	}

	//start timer for loading the file
	float startTime = hpt.TimeSinceLastCall();
	//file into buffer
	enWiki.read(buffer, giga);


	//end and print timer
	cout << "Loading file took: " << hpt.TimeSinceLastCall() - startTime << endl;
	//close
	free(buffer);
	enWiki.close();


	system("pause");
	return 0;
}