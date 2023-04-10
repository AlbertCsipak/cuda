#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

#define N 10
#define K 2

#define Threads 2000
#define Block_Size 5

int Block_Count = (N - 1) / Block_Size + 1;

int numbers[N] = {1,3,5,6,9,10,12,14,16,18};
int pos = -1;

__device__ int d_numbers[N];
__device__ int d_pos = -1;

//van e benne K hoszzu számtani sorozat
//cpun
//gpun 1 szállal
//gpun sok szállal
//shared memoryval
//blokkok n>1024

void CPU() {
	pos = -1;
	for (size_t i = 0; i < N-K; i++)
	{
		int counter = 0;
		while (counter < K && numbers[i + counter] + 2 == numbers[i + counter + 1]) {
			counter++;
		}
		if (counter == K) {
			pos = i;
		}
	}
}

__global__ void SingleThread() {
	d_pos = -1;
	for (size_t i = 0; i < N - K; i++)
	{
		int counter = 0;
		while (counter < K && d_numbers[i + counter] + 2 == d_numbers[i + counter + 1]) {
			counter++;
		}
		if (counter == K) {
			d_pos = i;
		}
	}
}

__global__ void MultiThread() {
	d_pos = -1;
	int counter = 0;
	while (counter < K && d_numbers[threadIdx.x + counter]+2 == d_numbers[threadIdx.x+counter +1]) {
		counter++;
	}
	if (counter == K) {
		d_pos = threadIdx.x;
	}

}

__global__ void MultiThreadShared() {
	d_pos = -1;
	__shared__ int s_numbers[N];
	
	if (threadIdx.x==0)
	{
		for (size_t i = 0; i < N; i++)
		{
			s_numbers[i] = d_numbers[i];
		}
	}
	__syncthreads();

	int counter = 0;
	while (counter < K && s_numbers[threadIdx.x + counter] + 2 == s_numbers[threadIdx.x + counter + 1]) {
		counter++;
	}
	if (counter == K) {
		d_pos = threadIdx.x;
	}

}

__global__ void MultiThreadWithBlocks() {
	d_pos = -1;

	int counter = 0;
	while (counter < K && d_numbers[blockIdx.x * blockDim.x + threadIdx.x + counter] + 2 == d_numbers[blockIdx.x * blockDim.x + threadIdx.x + counter + 1]) {
		counter++;
	}
	if (counter == K) {
		d_pos = blockIdx.x * blockDim.x + threadIdx.x;
	}

}

int main() {

	for (size_t i = 0; i < N; i++)
	{
		printf("[%d]=%d\n",i, numbers[i]);
	}

	cudaMemcpyToSymbol(d_numbers, numbers, N*sizeof(int));

	CPU();
	printf("\nCPU: A sorozat a %d. elemtol kezdodik", pos);
	pos = -1;

	SingleThread << <1, 1 >> > ();
	cudaMemcpyFromSymbol(&pos,d_pos,sizeof(int));
	printf("\nSingleThread: A sorozat a %d. elemtol kezdodik", pos);
	pos = -1;

	MultiThread << <1, N - K + 1 >> > ();
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int));
	printf("\nMultiThread: A sorozat a %d. elemtol kezdodik", pos);
	pos = -1;

	MultiThreadShared << <1, N-K+1>> > ();
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int));
	printf("\nMultiThreadShared: A sorozat a %d. elemtol kezdodik", pos);
	pos = -1;

	MultiThreadWithBlocks << <Block_Count, Block_Size >> > ();
	cudaMemcpyFromSymbol(&pos, d_pos, sizeof(int));
	printf("\nMultiThreadWithBlocks: A sorozat a %d. elemtol kezdodik", pos);
	pos = -1;

	cudaGetLastError();
}
