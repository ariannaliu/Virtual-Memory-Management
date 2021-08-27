#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <device_launch_parameters.h>
#include<math.h>
#include<math_functions.h>
#include<time.h>

// this cuda 原子锁 is copied from the internet
struct Lock
{
	int *mutex;
	__device__ Lock(void)
	{
#if __CUDA_ARCH__ >= 200
		mutex = new int;
		(*mutex) = 0;
#endif
	}
	__device__ ~Lock(void)
	{
#if __CUDA_ARCH__ >= 200
		delete mutex;
#endif
	}
	__device__ void lock(void)
	{
#if __CUDA_ARCH__ >= 200
		while (atomicCAS(mutex, 0, 1) != 0);
#endif
	}
	__device__ void unlock(void)
	{
#if __CUDA_ARCH__ >= 200
		atomicExch(mutex, 0);
#endif
	}
};


__device__ void init_invert_page_table(VirtualMemory *vm) {

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = 0x80000000;  //linked_list_prev
		vm->invert_page_table[i + 2 * (vm->PAGE_ENTRIES)] = 0x80000000; //linked_list_next
	}
	vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = 0x80000000; //head
	vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = 0x80000000; //tail
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
	u32 *invert_page_table, int *pagefault_num_ptr,
	int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
	int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
	int PAGE_ENTRIES) {
	// init variables
	vm->buffer = buffer;
	vm->storage = storage;
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;

	// init constants
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;

	// before first vm_write or vm_read
	init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {

	Lock myLock;
	int index = threadIdx.x;

	myLock.lock();

	/* Complate vm_read function to read single element from data buffer */
	uchar value;
	uint32_t page_num = addr / 32;
	uint32_t offset = addr % 32;
	uint32_t physical_addr;
	int next_index;
	int prev_index;
	int tail_index;
	int head_index;

	int flag_in = -1;  //mark the position that page number can be found
	int flag_ava = -1; //mark the first available page number position

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_num) {
			//we find the page number in the page table
			flag_in = i;
			break;
		}
	}
	if (flag_in != -1) {
		physical_addr = (flag_in) * 32 + offset;
		value = vm->buffer[physical_addr];
		head_index = vm->invert_page_table[3 * (vm->PAGE_ENTRIES)];
		if (vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] == flag_in) {
			//找到的东西是linked list的头，我们移到尾巴去
			next_index = vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)];
			vm->invert_page_table[next_index + vm->PAGE_ENTRIES] = 0x80000000;
			vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = next_index;

			tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
			vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = flag_in;
			vm->invert_page_table[flag_in + (vm->PAGE_ENTRIES)] = tail_index;
			vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
			vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = flag_in;
		}
		else if (vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] == flag_in) {
		}
		else {
			//我们找到的page是在中间
			prev_index = vm->invert_page_table[flag_in + vm->PAGE_ENTRIES];
			next_index = vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)];
			vm->invert_page_table[prev_index + 2 * (vm->PAGE_ENTRIES)] = next_index;
			vm->invert_page_table[next_index + vm->PAGE_ENTRIES] = prev_index;
			tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
			//把tail index的下一个指向我们的pivot
			vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = flag_in;
			//把pivot的下一个变成invalid
			vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
			vm->invert_page_table[flag_in + (vm->PAGE_ENTRIES)] = tail_index;
			vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = flag_in; //tail 变成 flag_in		
		}
	}
	else {
		//page is not find in the page table
		*vm->pagefault_num_ptr += 1;
		head_index = vm->invert_page_table[3 * (vm->PAGE_ENTRIES)];
		tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];

		//我们把东西都搬下去
		uint32_t secondary_addr = 32 * vm->invert_page_table[head_index];
		for (int n = 0; n < 32; n++) {
			vm->storage[secondary_addr + n] = vm->buffer[32 * head_index + n];
		}
		vm->invert_page_table[head_index] = page_num;
		next_index = vm->invert_page_table[head_index + 2 * (vm->PAGE_ENTRIES)];
		vm->invert_page_table[next_index + vm->PAGE_ENTRIES] = 0x80000000;
		vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = next_index;
		vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = head_index;
		vm->invert_page_table[head_index + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
		vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = head_index;

		for (int l = 0; l < 32; l++) {
			vm->buffer[32 * head_index + l] = vm->storage[32 * vm->invert_page_table[head_index] + l];
		}
		value = vm->buffer[32 * head_index + offset];
	}
	
	return value; //TODO
	myLock.unlock();
}






__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
	Lock myLock;
	int index = threadIdx.x;

	myLock.lock();
	/* Complete vm_write function to write value into data buffer */
	uint32_t page_num = addr / 32;
	uint32_t offset = addr % 32;
	uint32_t physical_addr;

	int flag_in = -1;  //mark the position that page number can be found
	int flag_ava = -1; //mark the first available page number position
	uint32_t next_index;
	uint32_t prev_index;
	uint32_t tail_index;
	uint32_t head_index;

	for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == page_num) {
			//we find the page number in the page table
			flag_in = i;
			break;
		}
	}
	if (flag_in != -1) {
		//page is in the page table
		physical_addr = (flag_in) * 32 + offset;
		vm->buffer[physical_addr] = value;

		//	  vm->invert_page_table[flag_in + 2*(vm->PAGE_ENTRIES)] = 0;
		if (vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] == flag_in && vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] == flag_in) {
			//如果我们找到的page entry是整个的头，头部尾部都是他

		}
		else if (vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] == flag_in) {
			//如果我们找到的page entry是整个的头，我们要把他移到尾部去
			//把头指向下一个
			next_index = vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)];
			vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = next_index; //头变成下一个index
			vm->invert_page_table[next_index + vm->PAGE_ENTRIES] = 0x80000000;
			tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
			//把tail index的下一个指向我们的pivot
			vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = flag_in;
			//把pivot的下一个变成invalid
			vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
			vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = flag_in; //tail 变成 flag_in		  
		}
		else if (vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] == flag_in) {
		}
		else {
			//我们找到的page是在中间
			prev_index = vm->invert_page_table[flag_in + vm->PAGE_ENTRIES];
			next_index = vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)];
			vm->invert_page_table[prev_index + 2 * (vm->PAGE_ENTRIES)] = next_index;
			vm->invert_page_table[next_index + vm->PAGE_ENTRIES] = prev_index;
			tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
			//把tail index的下一个指向我们的pivot
			vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = flag_in;
			//把pivot的下一个变成invalid
			vm->invert_page_table[flag_in + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
			vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = flag_in; //tail 变成 flag_in		  
		}
	}
	else {
		//page is not in the page table
		*vm->pagefault_num_ptr += 1;

		if (vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] == 0x80000000) {
			//新写入一个，这个东西在pagetable[0]的位置上
			vm->invert_page_table[0] = page_num;

			vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = uint32_t(0); //head
			vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = uint32_t(0); //tail
			physical_addr = 0 * 32 + offset;
			vm->buffer[physical_addr] = value;
		}
		else {
			for (int j = 0; j < vm->PAGE_ENTRIES; j++) {
				if (vm->invert_page_table[j] == 0x80000000) {
					flag_ava = j;      //we find the first empty entry at pos j
					break;
				}
			}
			if (flag_ava != -1) {
				vm->invert_page_table[flag_ava] = page_num;
				tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
				vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = flag_ava;
				vm->invert_page_table[flag_ava + (vm->PAGE_ENTRIES)] = tail_index;
				vm->invert_page_table[flag_ava + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
				vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = uint32_t(flag_ava);

				physical_addr = (flag_ava) * 32 + offset;
				vm->buffer[physical_addr] = value;

			}
			else {
				//the page table is full now, we have to replace the LRU
				head_index = vm->invert_page_table[3 * (vm->PAGE_ENTRIES)];
				tail_index = vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)];
				uint32_t secondary_addr = 32 * vm->invert_page_table[head_index];
				for (int l = 0; l < 32; l++) {
					vm->storage[secondary_addr + l] = vm->buffer[32 * head_index + l];
				}
				vm->invert_page_table[head_index] = page_num;
				vm->invert_page_table[3 * (vm->PAGE_ENTRIES)] = vm->invert_page_table[head_index + 2 * (vm->PAGE_ENTRIES)];
				next_index = vm->invert_page_table[head_index + 2 * (vm->PAGE_ENTRIES)];
				vm->invert_page_table[next_index + (vm->PAGE_ENTRIES)] = 0x80000000;
				vm->invert_page_table[tail_index + 2 * (vm->PAGE_ENTRIES)] = head_index;
				vm->invert_page_table[head_index + (vm->PAGE_ENTRIES)] = tail_index;
				vm->invert_page_table[head_index + 2 * (vm->PAGE_ENTRIES)] = 0x80000000;
				vm->invert_page_table[1 + 3 * (vm->PAGE_ENTRIES)] = head_index;
				vm->buffer[32 * head_index + offset] = value;
			}

		}
	}
	myLock.unlock();
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
	int input_size) {
	/* Complete snapshot function togther with vm_read to load elements from data
	 * to result buffer */
	for (int i = 0; i < input_size; i+=4) {
		results[i + threadIdx.x] = vm_read(vm, i + threadIdx.x);
		__syncthreads();
	}
}
