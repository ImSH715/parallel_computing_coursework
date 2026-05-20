#include "cuda.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Error Handler */
static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define GLIDER_BLOCK_X 16
#define GLIDER_BLOCK_Y 16

#define HISTOGRAM_THREADS 256
#ifndef HISTOGRAM_MAX_VALUE
#define HISTOGRAM_MAX_VALUE 128
#endif
#define HISTOGRAM_MAX_BINS (HISTOGRAM_MAX_VALUE + 1)

#define EMBOSS_BLOCK_X 16
#define EMBOSS_BLOCK_Y 16

/* Constant Memory */
__constant__ unsigned short d_glider_masks[16];
__constant__ float d_emboss_kernel[3][3];


/*  Glider helpers */
static unsigned short pattern_to_mask(const unsigned char pattern[3][3]) {
    unsigned short mask = 0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            if (pattern[y][x]) {
                mask |= (unsigned short)(1u << (y * 3 + x));
            }
        }
    }
    return mask;
}

static unsigned short rotate_mask_clockwise(unsigned short mask) {
    unsigned short rotated_mask = 0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            const int original_bit = y * 3 + x;
            const int nx = 2 - y;
            const int ny = x;
            const int new_bit = ny * 3 + nx;
            if (mask & (1u << original_bit)) {
                rotated_mask |= (unsigned short)(1u << new_bit);
            }
        }
    }
    return rotated_mask;
}

static unsigned short reflect_mask_horizontal(unsigned short mask) {
    unsigned short reflected_mask = 0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            const int original_bit = y * 3 + x;
            const int nx = 2 - x;
            const int ny = y;
            const int new_bit = ny * 3 + nx;
            if (mask & (1u << original_bit)) {
                reflected_mask |= (unsigned short)(1u << new_bit);
            }
        }
    }
    return reflected_mask;
}

static void build_glider_masks(unsigned short glider_masks[16]) {
    static const unsigned char GLIDER_1[3][3] = {
        {1, 0, 1},
        {1, 1, 0},
        {0, 0, 0}
    };
    static const unsigned char GLIDER_2[3][3] = {
        {0, 1, 0},
        {1, 0, 0},
        {1, 0, 1}
    };
    const unsigned short base_masks[2] = {
        pattern_to_mask(GLIDER_1),
        pattern_to_mask(GLIDER_2)
    };
    int idx = 0;
    for (int b = 0; b < 2; ++b) {
        for (int r = 0; r <= 1; ++r) {
            unsigned short m = r ? reflect_mask_horizontal(base_masks[b]) : base_masks[b];
            for (int rot = 0; rot < 4; ++rot) {
                glider_masks[idx++] = m;
                m = rotate_mask_clockwise(m);
            }
        }
    }
}

/* Count Gliders */

// =============================================================================
// FOR-LOOP VERSION of load_window_mask (for comparison / profiling)
// =============================================================================
__device__ __forceinline__
unsigned short load_window_mask_loop(const unsigned char cells_block[GLIDER_BLOCK_Y + 2][GLIDER_BLOCK_X + 2], int x, int y) {
    unsigned short window_mask = 0;
    for (int dy = 0; dy < 3; ++dy) {
        for (int dx = 0; dx < 3; ++dx) {
            if (cells_block[y + dy][x + dx]) {
                window_mask |= (unsigned short)(1u << (dy * 3 + dx));
            }
        }
    }
    return window_mask;
}

__global__ void count_gliders_kernel(const unsigned char* cells, int width, int height, unsigned long long* count) {
    __shared__ unsigned char cells_block[GLIDER_BLOCK_Y + 2][GLIDER_BLOCK_X + 2];
    __shared__ unsigned int block_count[GLIDER_BLOCK_Y * GLIDER_BLOCK_X];
    const int width2 = width - 2;
    const int height2 = height - 2;
    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
    const int local_idx = (int)(threadIdx.y * blockDim.x + threadIdx.x);

    // Load tile + halo into shared memory
    for (int load_y = (int)threadIdx.y; load_y < (int)blockDim.y + 2; load_y += (int)blockDim.y) {
        const int global_y = (int)(blockIdx.y * blockDim.y + load_y);
        for (int load_x = (int)threadIdx.x; load_x < (int)blockDim.x + 2; load_x += (int)blockDim.x) {
            const int global_x = (int)(blockIdx.x * blockDim.x + load_x);
            if (global_x < width && global_y < height) {
                cells_block[load_y][load_x] = cells[(size_t)global_y * (size_t)width + (size_t)global_x];
            }
            else {
                cells_block[load_y][load_x] = 0;
            }
        }
    }
    __syncthreads();

    unsigned int found = 0;
    if (x < width2 && y < height2) {
        // FOR-LOOP VERSION: swap the function call below to compare with unrolled version
        const unsigned short window_mask = load_window_mask_loop(cells_block, (int)threadIdx.x, (int)threadIdx.y);
        for (int i = 0; i < 16; ++i) {
            if (window_mask == d_glider_masks[i]) {
                found = 1;
                break;
            }
        }
    }

    block_count[local_idx] = found;
    __syncthreads();

    for (int stride = ((int)blockDim.x * (int)blockDim.y) / 2; stride > 0; stride >>= 1) {
        if (local_idx < stride) {
            block_count[local_idx] += block_count[local_idx + stride];
        }
        __syncthreads();
    }

    if (local_idx == 0 && block_count[0] > 0) {
        atomicAdd(count, (unsigned long long)block_count[0]);
    }
}

uint64_t cuda_countGliders(const unsigned char* cells, const size_t width, const size_t height) {
    if (cells == NULL || width < 3 || height < 3) {
        return 0;
    }

    unsigned char* d_cells = NULL;
    unsigned long long* d_glider_count = NULL;
    unsigned long long h_glider_count = 0;

    unsigned short glider_masks[16];
    build_glider_masks(glider_masks);

    cuda_check(cudaMalloc((void**)&d_cells, width * height * sizeof(unsigned char)), "cudaMalloc d_cells");
    cuda_check(cudaMalloc((void**)&d_glider_count, sizeof(unsigned long long)), "cudaMalloc d_glider_count");

    cuda_check(cudaMemcpy(d_cells, cells, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy d_cells");
    cuda_check(cudaMemcpyToSymbol(d_glider_masks, glider_masks, sizeof(glider_masks)), "cudaMemcpyToSymbol d_glider_masks");
    cuda_check(cudaMemset(d_glider_count, 0, sizeof(unsigned long long)), "cudaMemset d_glider_count");

    const dim3 block(GLIDER_BLOCK_X, GLIDER_BLOCK_Y);
    const dim3 grid(
        (unsigned int)((width - 2 + block.x - 1) / block.x),
        (unsigned int)((height - 2 + block.y - 1) / block.y)
    );

    count_gliders_kernel << <grid, block >> > (d_cells, (int)width, (int)height, d_glider_count);
    cuda_check(cudaGetLastError(), "count_gliders_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "count_gliders_kernel sync");

    cuda_check(cudaMemcpy(&h_glider_count, d_glider_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost),
        "cudaMemcpy h_glider_count");

    cudaFree(d_cells);
    cudaFree(d_glider_count);

    return (uint64_t)h_glider_count;
}

/* Histogram */
__global__ void histogram_kernel(const int* numbers, int length, int bin_width, int histogram_len, int* histogram) {
    __shared__ int histogram_block[HISTOGRAM_MAX_BINS];

    for (int bin = (int)threadIdx.x; bin < histogram_len; bin += (int)blockDim.x) {
        histogram_block[bin] = 0;
    }
    __syncthreads();

    const int index = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(blockDim.x * gridDim.x);

    for (int i = index; i < length; i += stride) {
        const int value = numbers[i];
        if (value >= 0 && value <= HISTOGRAM_MAX_VALUE) {
            int bin = value / bin_width;
            if (bin >= histogram_len) {
                bin = histogram_len - 1;
            }
            atomicAdd(&histogram_block[bin], 1);
        }
    }
    __syncthreads();

    for (int bin = (int)threadIdx.x; bin < histogram_len; bin += (int)blockDim.x) {
        if (histogram_block[bin] > 0) {
            atomicAdd(&histogram[bin], histogram_block[bin]);
        }
    }
}

size_t cuda_histogram(const int* numbers, size_t length, int bin_width, int* histogram_out) {
    if (bin_width <= 0) {
        return 0;
    }

    const size_t histogram_len = (size_t)ceil((double)HISTOGRAM_MAX_VALUE / (double)bin_width);

    if (histogram_out == NULL) {
        return histogram_len;
    }

    memset(histogram_out, 0, sizeof(int) * histogram_len);

    if (numbers == NULL || length == 0) {
        return histogram_len;
    }

    int* d_numbers = NULL;
    int* d_histogram = NULL;

    const size_t numbers_bytes = sizeof(int) * length;
    const size_t histogram_bytes = sizeof(int) * histogram_len;

    cuda_check(cudaMalloc((void**)&d_numbers, numbers_bytes), "cudaMalloc d_numbers");
    cuda_check(cudaMalloc((void**)&d_histogram, histogram_bytes), "cudaMalloc d_histogram");

    cuda_check(cudaMemcpy(d_numbers, numbers, numbers_bytes, cudaMemcpyHostToDevice), "cudaMemcpy d_numbers");
    cuda_check(cudaMemset(d_histogram, 0, histogram_bytes), "cudaMemset d_histogram");

    int blocks = (int)((length + HISTOGRAM_THREADS - 1) / HISTOGRAM_THREADS);
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    histogram_kernel <<<blocks, HISTOGRAM_THREADS >>> (d_numbers, (int)length, bin_width, (int)histogram_len, d_histogram);

    cuda_check(cudaGetLastError(), "histogram_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "histogram_kernel sync");

    cuda_check(cudaMemcpy(histogram_out, d_histogram, histogram_bytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy histogram_out");

    cudaFree(d_numbers);
    cudaFree(d_histogram);

    return histogram_len;
}

/* Emboss */
__global__ void emboss_kernel(const unsigned char* pixels, int width, int height, unsigned char* output) {
    __shared__ unsigned char pixels_block[(EMBOSS_BLOCK_Y + 2) * (EMBOSS_BLOCK_X + 2) * 3];

    const int out_w = width - 2;
    const int out_h = height - 2;

    const int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);

    const int block_width = (int)blockDim.x + 2;

    for (int local_y = (int)threadIdx.y; local_y < (int)blockDim.y + 2; local_y += (int)blockDim.y) {
        for (int local_x = (int)threadIdx.x; local_x < (int)blockDim.x + 2; local_x += (int)blockDim.x) {
            const int global_x = (int)(blockIdx.x * blockDim.x + local_x);
            const int global_y = (int)(blockIdx.y * blockDim.y + local_y);
            const int block_index = ((local_y * block_width) + local_x) * 3;

            if (global_x < width && global_y < height) {
                const int pixel_index = ((global_y * width) + global_x) * 3;
                pixels_block[block_index + 0] = pixels[pixel_index + 0];
                pixels_block[block_index + 1] = pixels[pixel_index + 1];
                pixels_block[block_index + 2] = pixels[pixel_index + 2];
            }
            else {
                pixels_block[block_index + 0] = 0;
                pixels_block[block_index + 1] = 0;
                pixels_block[block_index + 2] = 0;
            }
        }
    }
    __syncthreads();

    if (x >= out_w || y >= out_h) {
        return;
    }

    float pixel_sum = 0.0f;
    for (int ky = 0; ky < 3; ++ky) {
        for (int kx = 0; kx < 3; ++kx) {
            const int block_index = (((int)threadIdx.y + ky) * block_width + ((int)threadIdx.x + kx)) * 3;
            const float r = (float)pixels_block[block_index + 0];
            const float g = (float)pixels_block[block_index + 1];
            const float b = (float)pixels_block[block_index + 2];
            const float gray = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            pixel_sum += gray * d_emboss_kernel[ky][kx];
        }
    }

    pixel_sum += 128.0f;
    if (pixel_sum < 0.0f) pixel_sum = 0.0f;
    if (pixel_sum > 255.0f) pixel_sum = 255.0f;

    output[(size_t)y * (size_t)out_w + (size_t)x] = (unsigned char)pixel_sum;
}

void cuda_emboss(const unsigned char* pixels, const size_t width, const size_t height, unsigned char* output) {
    static const float EMBOSS_KERNEL[3][3] = {
        {-2.0f, -1.0f, 0.0f},
        {-1.0f,  0.0f, 1.0f},
        { 0.0f,  1.0f, 2.0f}
    };

    if (pixels == NULL || output == NULL || width < 3 || height < 3) {
        return;
    }

    const int out_w = (int)(width - 2);
    const int out_h = (int)(height - 2);

    unsigned char* d_pixels = NULL;
    unsigned char* d_output = NULL;

    cuda_check(cudaMalloc((void**)&d_pixels, width * height * 3u * sizeof(unsigned char)), "cudaMalloc d_pixels");
    cuda_check(cudaMalloc((void**)&d_output, (size_t)out_w * (size_t)out_h * sizeof(unsigned char)), "cudaMalloc d_output");

    cuda_check(cudaMemcpy(d_pixels, pixels, width * height * 3u * sizeof(unsigned char), cudaMemcpyHostToDevice), "cudaMemcpy d_pixels");
    cuda_check(cudaMemcpyToSymbol(d_emboss_kernel, EMBOSS_KERNEL, sizeof(EMBOSS_KERNEL)), "cudaMemcpyToSymbol d_emboss_kernel");

    const dim3 block(EMBOSS_BLOCK_X, EMBOSS_BLOCK_Y);
    const dim3 grid(
        (unsigned int)((out_w + block.x - 1) / block.x),
        (unsigned int)((out_h + block.y - 1) / block.y)
    );

    emboss_kernel << <grid, block >> > (d_pixels, (int)width, (int)height, d_output);
    cuda_check(cudaGetLastError(), "emboss_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "emboss_kernel sync");

    cuda_check(cudaMemcpy(output, d_output, (size_t)out_w * (size_t)out_h * sizeof(unsigned char), cudaMemcpyDeviceToHost), "cudaMemcpy output");

    cudaFree(d_pixels);
    cudaFree(d_output);
}
