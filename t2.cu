#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

//occupancy = 67%%

//n tömb k konstans 

//1. feladat: melyik k hosszu rész sorozat összege a legkisebb ,összeg és index 
//2. feladat shared memory
//3. feladat mi az ideális blokkméret - max blokkméret lekérdezése
//4. feladat: ciklussal időmérés
//5. feladat occupancy calculatorral ->hatékonyság 

#define N 10
#define K 3

int numbers[N] = { 1,5,1,6,9,0,1,3,1,1 };
int pos;
int minSum;

__device__ int d_numbers[N];
__device__ int d_pos;
__device__ int d_minSum;

void CPU() {
	pos = -1;
	minSum = 100;
	int counter = 0;
	int tmpSum = 0;
	for (size_t i = 0; i < N-K; i++)
	{
		counter = 0;
		tmpSum = 0;
		while (counter < K) {
			tmpSum += numbers[counter+i];
			counter++;
			;
		}
		if (tmpSum < minSum)
		{
			pos = i;
			minSum = tmpSum;
		}
	}
}

__global__ void SingleThread() {
	d_pos = -1;
	d_minSum = 100;
	int counter = 0;
	int tmpSum = 0;
	for (size_t i = 0; i < N - K; i++)
	{
		counter = 0;
		tmpSum = 0;
		while (counter < K) {
			tmpSum += d_numbers[counter + i];
			counter++;
			;
		}
		if (tmpSum < d_minSum)
		{
			d_pos = i;
			d_minSum = tmpSum;
		}
	}
}

__global__ void MultiThread() {
	d_pos = -1;
	d_minSum = 100;

	int counter = 0;
	int tmpSum = 0;
	while (counter < K) {
		tmpSum += d_numbers[counter + threadIdx.x];
		counter++;
	}
	atomicMin(&d_minSum, tmpSum);

	if (d_minSum == tmpSum)
	{
		d_pos = threadIdx.x;
	}
}

__global__ void MultiThreadShared() {
	d_pos = -1;
	d_minSum = 100;
	__shared__ int s_numbers[N];
	__shared__ int s_minSum;
	int counter = 0;
	int tmpSum = 0;

	if (threadIdx.x == 0)
	{
		for (size_t i = 0; i < N; i++)
		{
			s_numbers[i] = d_numbers[i];
		}
		s_minSum = d_minSum;
	}
	__syncthreads();

	while (counter < K) {
		tmpSum += s_numbers[counter + threadIdx.x];
		counter++;
	}
	
	atomicMin(&s_minSum, tmpSum);

	if (s_minSum == tmpSum)
	{
		d_pos = threadIdx.x;
		d_minSum = s_minSum;
	}

}

__global__ void MultiThreadWithBlocks() {
	d_pos = -1;
	d_minSum = 100;
	__shared__ int s_numbers[N];
	__shared__ int s_minSum;
	int counter = 0;
	int tmpSum = 0;

	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	{
		for (size_t i = 0; i < N; i++)
		{
			s_numbers[i] = d_numbers[i];
		}
		s_minSum = d_minSum;
	}
	__syncthreads();

	while (counter < K) {
		tmpSum += s_numbers[counter + blockIdx.x * blockDim.x + threadIdx.x];
		counter++;
	}

	atomicMin(&s_minSum, tmpSum);

	if (s_minSum == tmpSum)
	{
		d_pos = blockIdx.x * blockDim.x + threadIdx.x;
		d_minSum = s_minSum;
	}
}

int main() {

	for (size_t i = 0; i < N; i++)
	{
		printf("[%d]=%d\n", i, numbers[i]);
	}

	//ideális blokkméret
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("\nMax Threads Per Block: %d", prop.maxThreadsPerBlock);
	printf("\nMax Blocks: %d", prop.maxBlocksPerMultiProcessor);

	//cpu
	CPU();
	printf("\n\nCPU: A sorozat a %d. elemtol kezdodik, osszege: %d", pos,minSum);

	//Single Thread
	cudaMemcpyToSymbol(d_numbers, numbers, N * sizeof(int)); //cpy to gpu
	SingleThread << <1, 1 >> > ();
	cudaMemcpyFromSymbol(&minSum, d_minSum, sizeof(int)); //cpy back
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int)); //cpy back
	printf("\nSingleThread: A sorozat a %d. elemtol kezdodik, osszege: %d", pos, minSum);

	//MultiThread
	cudaMemcpyToSymbol(d_numbers, numbers, N * sizeof(int)); //cpy to gpu
	MultiThread << <1, N - K  >> > ();
	cudaMemcpyFromSymbol(&minSum, d_minSum, sizeof(int)); //cpy back
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int)); //cpy back
	printf("\nMultiThread: A sorozat a %d. elemtol kezdodik, osszege: %d", pos, minSum);

	//MultiThreadShared
	cudaMemcpyToSymbol(d_numbers, numbers, N * sizeof(int)); //cpy to gpu
	MultiThreadShared << <1, N - K >> > ();
	cudaMemcpyFromSymbol(&minSum, d_minSum, sizeof(int)); //cpy back
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int)); //cpy back
	printf("\nMultiThreadShared: A sorozat a %d. elemtol kezdodik, osszege: %d\n", pos, minSum);

	//MultiThreadWithBlocks
	for (size_t i = 1; i <= prop.maxBlocksPerMultiProcessor; i++)
	{
		int Block_Count = (N - 1) /i + 1;

		cudaMemcpyToSymbol(d_numbers, numbers, N * sizeof(int)); //cpy to gpu
		MultiThreadWithBlocks << <Block_Count, N-K >> > ();
		cudaMemcpyFromSymbol(&minSum, d_minSum, sizeof(int)); //cpy back
		cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int)); //cpy back
		printf("\nMultiThreadWithBlocks: A sorozat a %d. elemtol kezdodik, osszege: %d, blocksize: %d", pos, minSum,i);
	}

	printf("\n");

	cudaGetLastError();
}
