﻿#ifndef VIRTUAL_MEMORY_H
#define VIRTUAL_MEMORY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <inttypes.h>

typedef unsigned char uchar;
typedef uint32_t u32;

struct VirtualMemory
{
  uchar *buffer;
  uchar *storage;
  u32 *invert_page_table;
  int *pagefault_num_ptr;
  int PAGESIZE;
  int INVERT_PAGE_TABLE_SIZE;
  int PHYSICAL_MEM_SIZE;
  int STORAGE_SIZE;
  int PAGE_ENTRIES;
};

__device__ void convert_to_invalid(VirtualMemory* vm, u32 frame_number);

__device__ int search(VirtualMemory* vm, u32 page_number);

__device__ void copy_data(VirtualMemory* vm, u32 frame_number, u32 page_number);

__device__ bool page_fault(VirtualMemory* vm, u32 page_number);

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage, u32 *invert_page_table, int *pagefault_num_ptr, int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int PHYSICAL_MEM_SIZE, int STORAGE_SIZE, int PAGE_ENTRIES);

__device__ uchar vm_read(VirtualMemory *vm, u32 addr);

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value);

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size);

#endif
