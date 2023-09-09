#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__device__ void copy_data(VirtualMemory* vm, u32 page_number, u32 frame_number)
{
	for (int i = 0; i < 32; i++)
	{
		vm->storage[vm->invert_page_table[frame_number] * 32 + i] = vm->buffer[frame_number * 32 + i];
		vm->buffer[frame_number * 32 + i] = vm->storage[page_number * 32 + i];
	}
	vm->invert_page_table[frame_number] = page_number;
}

__device__ bool page_fault(VirtualMemory* vm, u32 page_number)
{
	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
		if (vm->invert_page_table[i] == page_number)
			return false;
	return true;
}

__device__ void invalid(VirtualMemory* vm, u32 frame_number)
{
	int head, tmp;
	bool tag = true;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == frame_number)
		{
			tag = false;
			head = i;
		}
	if (tag)
		return;
	tmp = vm->invert_page_table[vm->PAGE_ENTRIES + head];
	for (int i = vm->PAGE_ENTRIES + head; i < 2 * vm->PAGE_ENTRIES - 1; i++)
		vm->invert_page_table[i] = vm->invert_page_table[i + 1];
	vm->invert_page_table[2 * vm->PAGE_ENTRIES - 1] = tmp;
}

__device__ int search(VirtualMemory* vm, u32 page_number)
{
	u32 frame_number = -1;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
		if (vm->invert_page_table[i] == page_number)
		{
			frame_number = i;
			break;
		}
	return frame_number;
}

__device__ void vm_init(VirtualMemory* vm, uchar* buffer, uchar* storage, u32* invert_page_table, int* pagefault_num_ptr,
	int PAGESIZE, int INVERT_PAGE_TABLE_SIZE, int PHYSICAL_MEM_SIZE, int STORAGE_SIZE, int PAGE_ENTRIES)
{
	vm->buffer = buffer;
	vm->storage = storage;
	vm->invert_page_table = invert_page_table;
	vm->pagefault_num_ptr = pagefault_num_ptr;
	vm->PAGESIZE = PAGESIZE;
	vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
	vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
	vm->STORAGE_SIZE = STORAGE_SIZE;
	vm->PAGE_ENTRIES = PAGE_ENTRIES;
	for (int i = 0; i < vm->PAGE_ENTRIES; i++)
	{
		vm->invert_page_table[i] = 0x80000000;
		vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
	}
}

__device__ uchar vm_read(VirtualMemory* vm, u32 addr)
{
	u32 page_number = addr / 32, frame_number;
	if (!page_fault(vm, page_number))
		frame_number = search(vm, page_number);
	else
	{
		*vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
		frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
		copy_data(vm, page_number, frame_number);
	}
	invalid(vm, frame_number);
}

__device__ void vm_write(VirtualMemory* vm, u32 addr, uchar value)
{
	u32 page_number = addr / 32, frame_number, page_number_storage;
	if (!page_fault(vm, page_number))
		frame_number = search(vm, page_number);
	else
	{
		*vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
		frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
		if (vm->invert_page_table[frame_number] != 0x80000000)
		{
			page_number_storage = vm->invert_page_table[frame_number];
			for (int i = 0; i < 32; i++)
				vm->storage[page_number_storage * 32 + i] = vm->buffer[frame_number * 32 + i];
		}
		vm->invert_page_table[frame_number] = page_number;
	}
	vm->buffer[frame_number * 32 + addr % 32] = value;
	invalid(vm, frame_number);
}

__device__ void vm_snapshot(VirtualMemory* vm, uchar* results, int offset, int input_size)
{
	for (int i = 0; i < input_size; i++)
	{
		u32 page_number = i / 32, frame_number;
		if (!page_fault(vm, page_number))
			frame_number = search(vm, page_number);
		else
		{
			*vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
			frame_number = vm->invert_page_table[vm->PAGE_ENTRIES];
			copy_data(vm, page_number, frame_number);
		}
		for (int j = 0; j < 32; j++)
			results[page_number * 32 + j] = vm->buffer[frame_number * 32 + j];
		invalid(vm, frame_number);
	}
}
