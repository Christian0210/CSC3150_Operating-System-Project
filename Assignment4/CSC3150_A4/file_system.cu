#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 gtime_create = 0;
__device__ __managed__ u32 block_position = 0;
__device__ __managed__ u32 FCB_position = 4096;
__device__ __managed__ u32 current_FCB_position = 4096;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;
  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;
}

__device__ __managed__ struct FCB_node
{
	char name[20];
	u32 date;
	u32 size;
} node;


__device__ void segment_management(FileSystem *fs, u32 fp, u32 original_size) 
{
	u32 position = fs->FILE_BASE_ADDRESS + fp * 32;
	u32 size = ((original_size - 1) / 32 + 1) * 32;
	while ((fs->volume[position + size] != 0 || (position + size) %32 != 0)&& position + original_size < fs->STORAGE_SIZE)
	{
		fs->volume[position] = fs->volume[position + size];
		fs->volume[position + size] = 0;
		position++;
	}
	for (int i = 0; i < block_position / 8 + 1; i++)
		fs->volume[i] = 0;
	block_position = block_position - (original_size - 1) / 32 - 1;
	u32 whole_block = block_position / 8;
	u32 remainder = block_position % 8;
	for (int i = 0; i < whole_block && i < fs->SUPERBLOCK_SIZE ; i++)
		fs->volume[i] = 511;
	for (int i = 0; i < remainder; i++)
		fs->volume[whole_block] = fs->volume[whole_block] + (1 << i);
	u32 FCB_block_position;
	for (int i = 4096; i < 36863; i = i + 32)
	{
		if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i + 3] == 0) break;
		FCB_block_position = (fs->volume[i + 28] << 24) + (fs->volume[i + 29] << 16) + (fs->volume[i + 30] << 8) + (fs->volume[i + 31]);
		if (FCB_block_position > fp) 
		{
			FCB_block_position = FCB_block_position - (original_size - 1) / 32 - 1;
			fs->volume[i + 28] = FCB_block_position >> 24;
			fs->volume[i + 29] = FCB_block_position >> 16;
			fs->volume[i + 30] = FCB_block_position >> 8;
			fs->volume[i + 31] = FCB_block_position;
		}
	}
}

__device__ void output(FileSystem*fs, u32 stop_position, int op) 
{
	char name[20];
	if (op) 
	{
		u32 size;
		printf("===sort by file size===\n");
		for (u32 i = 4096; i <= stop_position; i += 32)
		{
			for (int j = 4; j < 24; j++)
				name[j - 4] = fs->volume[i + j];
			size = (fs->volume[i] << 24) + (fs->volume[i + 1] << 16) + (fs->volume[i + 2] << 8) + (fs->volume[i + 3]);
			printf("%s %d\n", name, size);
		}
	}
	else 
	{
		printf("===sort by modified time===\n");
		for (u32 i = 4096; i <= stop_position; i += 32)
		{
			for (int j = 4; j < 24; j++)
				name[j - 4] = fs->volume[i + j];
			printf("%s\n", name);
		}
	}
}


__device__ void swap(FileSystem* fs, u32 a, u32 b)
{
	for (int i = 0; i < 32; i++)
	{
		uchar tmp = fs->volume[a + i];
		fs->volume[a + i] = fs->volume[b + i];
		fs->volume[b + i] = tmp;
	}
}


__device__ void filesort(FileSystem *fs, u32 left, u32 right, int op)
{
	if (op != 0) 
	{
		for (int i = left; i < right; i = i + 32)
		{
			u32 maxsize = (fs->volume[i] << 24) + (fs->volume[i + 1] << 16) + (fs->volume[i + 2] << 8) + (fs->volume[i + 3]);
			u32 maxsize_date = (fs->volume[i + 24] << 8) + (fs->volume[i + 25]);
			int tag = i;
			for (int j = i; j <= right; j = j + 32)
			{
				u32 size = (fs->volume[j] << 24) + (fs->volume[j + 1] << 16) + (fs->volume[j + 2] << 8) + (fs->volume[j + 3]);
				u32 date = (fs->volume[j + 24] << 8) + (fs->volume[j + 25]);
				if ((size > maxsize) || (size == maxsize && maxsize_date > date))
				{
					tag = j;
					maxsize = size;
					maxsize_date = date;
				}
			}
			swap(fs, i, tag);
		}
	}
	else 
	{
		for (int i = left; i < right; i = i + 32)
		{
			u32 maxdate = (fs->volume[i + 26] << 8) + (fs->volume[i + 27]);
			int tag = i;
			for (int j = i; j <= right; j = j + 32)
			{
				u32 date = (fs->volume[j + 26] << 8) + (fs->volume[j + 27]);
				if (date > maxdate)
				{
					tag = j;
					maxdate = date;
				}
			}
			swap(fs, i, tag);
		}
	}
}

__device__ u32 check(FileSystem *fs, char *s) 
{
	int i, j, flag;
	for (i = 4096; i < 36863; i = i + 32)
	{
		flag = 0;
		if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i + 3] == 0)
			break;
		for (j = 4; j < 24; j++) 
		{
			if (fs->volume[i + j] != s[j - 4]) 
			{
				flag = 1;
				break;
			}
		}
		if (flag == 0) 
			return i;
	}
	return -1;
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	int i;
	if (check(fs, s) == -1) 
	{
		if (!op) 
		{
			printf("File Open Failed\n");
			return -1;
		}
		current_FCB_position = FCB_position;
		for (i = 4; i < 24; i++)
			fs->volume[FCB_position + i] = s[i - 4];
		fs->volume[FCB_position + i] = gtime_create >> 8;
		fs->volume[FCB_position + i + 1] = gtime_create;
		fs->volume[FCB_position + i + 2] = gtime >> 8;
		fs->volume[FCB_position + i + 3] = gtime;
		fs->volume[FCB_position + i + 4] = block_position >> 24;
		fs->volume[FCB_position + i + 5] = block_position >> 16;
		fs->volume[FCB_position + i + 6] = block_position >> 8;
		fs->volume[FCB_position + i + 7] = block_position;
		gtime_create++;
		gtime++;
		FCB_position = FCB_position + 32;
		return block_position;
	}
	else 
	{
		current_FCB_position = check(fs, s);
		u32 start_block = (fs->volume[current_FCB_position + 28] << 24) + (fs->volume[current_FCB_position + 29] << 16) 
			+ (fs->volume[current_FCB_position + 30] << 8) + (fs->volume[current_FCB_position + 31]);
		if (op == 1) 
		{
			u32 size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16)
				+ (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
			for (i = 0; i < size; i++)
				fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
			for (i = 0; i < (size - 1) / 32 + 1; i++)
			{
				u32 sblock = start_block + i;
				int snum = sblock % 8;
				fs->volume[sblock / 8] = fs->volume[sblock / 8] - (1 << snum);
			}
			fs->volume[current_FCB_position + 26] = gtime >> 8;
			fs->volume[current_FCB_position + 27] = gtime;
			gtime++;
		}
		return start_block;
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	for (int i = 0; i < size; i++)
		output[i] = fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS];
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	int i;
	if ((fs->volume[(fp + (size - 1) / 32)/8] >> (fp + (size - 1) / 32) % 8) % 2 == 0)
	{
		u32 old_file_size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
		u32 original_size = old_file_size - size;
		for (i = 0; i < size; i++) 
		{
			fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];
			if (i % 32 == 0)
			{ 
				u32 sblock = fp + i / 32;
				int snum = sblock % 8;
				fs->volume[sblock / 8] = fs->volume[sblock / 8] + (1 << snum);
			}
		}
		if (int(original_size) < 0) 
			block_position = block_position + (-original_size - 1) / 32 + 1;
		for (i = 0; i < 4; i++) 
			fs->volume[current_FCB_position + i] = size >> (3 - i) * 8;
		if (original_size > 0 && old_file_size != 0 && fp != block_position - 1)
			segment_management(fs, fp + (size - 1) / 32 + 1, original_size);
	}
	else 
	{
		u32 original_size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
		if (block_position * 32 - 1 + size >= fs->SUPERBLOCK_SIZE)
			return -1;
		else 
		{
			for (i = 0; i < size; i++)
			{ 
				fs->volume[block_position * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];
				if (i % 32 == 0) 
				{
					u32 sblock = block_position + i / 32;
					int snum = sblock % 8;
					fs->volume[sblock / 8] = fs->volume[sblock / 8] + (1 << snum);
				}
			}
			for (i = 0; i < 4; i++)
				fs->volume[current_FCB_position + i] = size >> (3 - i) * 8;
			fs->volume[current_FCB_position + 28] = block_position >> 16;
			fs->volume[current_FCB_position + 29] = block_position >> 16;
			fs->volume[current_FCB_position + 30] = block_position >> 8;
			fs->volume[current_FCB_position + 31] = block_position;
		}
		segment_management(fs, fp, original_size);
	}
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	u32 end;
	for (u32 i = 4096; i < 36895; i = i + 32)
	{
		u32 size = (fs->volume[i] << 24) + (fs->volume[i + 1] << 16) + (fs->volume[i + 2] << 8) + (fs->volume[i + 3]);
		end = i - 32;
		if (!size)
		{
			size = (fs->volume[4096] << 24) + (fs->volume[4096 + 1] << 16) + (fs->volume[4096 + 2] << 8) + (fs->volume[4096 + 3]);
			break;
		}
	}
	if (end < 4096)
		return;
	filesort(fs, 4096, end, op);
	output(fs, end, op);
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	int i, j;
	if (check(fs, s) == -1)
		return;
	current_FCB_position = check(fs, s);
	u32 start_block = (fs->volume[current_FCB_position + 28] << 24) + (fs->volume[current_FCB_position + 29] << 16) + (fs->volume[current_FCB_position + 30] << 8) + (fs->volume[current_FCB_position + 31]);
	u32 size = (fs->volume[current_FCB_position] << 24) + (fs->volume[current_FCB_position + 1] << 16) + (fs->volume[current_FCB_position + 2] << 8) + (fs->volume[current_FCB_position + 3]);
	for (i = 0; i < size; i++)
		fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
	for (i = 0; i < (size - 1) / 32 + 1; i++)
		fs->volume[start_block + i] = 0;
	for (i = 0; i < 32; i++)
		fs->volume[current_FCB_position + i] = 0;
	segment_management(fs, start_block, size);
	for (i = current_FCB_position; i < 36863; i = i + 32)
	{
		if (fs->volume[i + 32] == 0 && fs->volume[i + 32 + 1] == 0 && fs->volume[i + 32 + 2] == 0 && fs->volume[i + 32 + 3] == 0) 
			break;
		for (j = 0; j < 32; j++) 
		{
			fs->volume[i + j] = fs->volume[i + j + 32];
			fs->volume[i + j + 32] = 0;
		}
	}
	FCB_position = FCB_position - 32;
}
