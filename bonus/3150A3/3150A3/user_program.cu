#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__device__ void user_program(VirtualMemory *vm, uchar *input, uchar *results,
                             int input_size) {
//  int i = threadIdx.x;
//  printf("This thread has id %d",i);

	for (int i = 0; i < input_size; i += 4) {
		vm_write(vm, i + threadIdx.x, input[i + threadIdx.x]);
		__syncthreads();
	}
    

	for (int i = input_size - 1; i >= input_size - 32769; i -= 4) {
		int value = vm_read(vm, i);
		__syncthreads();
	}
    

  vm_snapshot(vm, results, 0, input_size);
  __syncthreads();
}
