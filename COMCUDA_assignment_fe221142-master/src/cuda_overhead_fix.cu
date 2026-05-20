#include "cuda.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

#define GLIDER_BLOCK_X 16
#define GLIDER_BLOCK_Y 16
#define HISTOGRAM_THREADS 256
#define HISTOGRAM_MAX_BINS (HISTOGRAM_MAX_VALUE + 1)
#define EMBOSS_BLOCK_X 16
#define EMBOSS_BLOCK_Y 16

__constant__ unsigned short c_glider_masks[16];
__constant__ float c_emboss_kernel[3][3];

/* Reusable device buffers */
static unsigned char* cached_cells = NULL;
static size_t cached_cells_bytes = 0;

static unsigned long long* cached_count = NULL;
static size_t cached_count_bytes = 0;

static int* cached_numbers = NULL;
static size_t cached_numbers_bytes = 0;

static int* cached_histogram = NULL;
static size_t cached_histogram_bytes = 0;

static unsigned char* cached_pixels = NULL;
static size_t cached_pixels_bytes = 0;

static unsigned char* cached_output = NULL;
static size_t cached_output_bytes = 0;

static void ensure_u8_buffer(unsigned char** ptr, size_t* capacity, size_t required, const char* name) {
    if (*capacity < required) {
        if (*ptr != NULL) {
            cudaFree(*ptr);
        }
        cuda_check(cudaMalloc((void**)ptr, required), name);
        *capacity = required;
    }
}

static void ensure_int_buffer(int** ptr, size_t* capacity, size_t required, const char* name) {
    if (*capacity < required) {
        if (*ptr != NULL) {
            cudaFree(*ptr);
        }
        cuda_check(cudaMalloc((void**)ptr, required), name);
        *capacity = required;
    }
}

static void ensure_ull_buffer(unsigned long long** ptr, size_t* capacity, size_t required, const char* name) {
    if (*capacity < required) {
        if (*ptr != NULL) {
            cudaFree(*ptr);
        }
        cuda_check(cudaMalloc((void**)ptr, required), name);
        *capacity = required;
    }
}

/* Glider helpers */
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
    unsigned short rotated = 0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            const int old_bit = y * 3 + x;
            const int nx = 2 - y;
            const int ny = x;
            const int new_bit = ny * 3 + nx;
            if (mask & (1u << old_bit)) {
                rotated |= (unsigned short)(1u << new_bit);
            }
        }
    }
    return rotated;
}

static unsigned short reflect_mask_horizontal(unsigned short mask) {
    unsigned short reflected = 0;
    for (int y = 0; y < 3; ++y) {
        for (int x = 0; x < 3; ++x) {
            const int old_bit = y * 3 + x;
            const int nx = 2 - x;
            const int new_bit = y * 3 + nx;
            if (mask & (1u << old_bit)) {
                reflected |= (unsigned short)(1u << new_bit);
            }
        }
    }
    return reflected;
}

static void build_glider_masks(unsigned short masks[16]) {
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
        for (int reflect = 0; reflect <= 1; ++reflect) {
            unsigned short m = reflect ? reflect_mask_horizontal(base_masks[b]) : base_masks[b];
            for (int rot = 0; rot < 4; ++rot) {
                masks[idx++] = m;
                m = rotate_mask_clockwise(m);
            }
        }
    }
}

/* Count Gliders */
__device__ __forceinline__ unsigned short window_mask_from_shared(
    const unsigned char tile[GLIDER_BLOCK_Y + 2][GLIDER_BLOCK_X + 2],
    int tx,
    int ty) {
    unsigned short mask = 0;
    if (tile[ty + 0][tx + 0]) mask |= (unsigned short)(1u << 0);
    if (tile[ty + 0][tx + 1]) mask |= (unsigned short)(1u << 1);
    if (tile[ty + 0][tx + 2]) mask |= (unsigned short)(1u << 2);
    if (tile[ty + 1][tx + 0]) mask |= (unsigned short)(1u << 3);
    if (tile[ty + 1][tx + 1]) mask |= (unsigned short)(1u << 4);
    if (tile[ty + 1][tx + 2]) mask |= (unsigned short)(1u << 5);
    if (tile[ty + 2][tx + 0]) mask |= (unsigned short)(1u << 6);
    if (tile[ty + 2][tx + 1]) mask |= (unsigned short)(1u << 7);
    if (tile[ty + 2][tx + 2]) mask |= (unsigned short)(1u << 8);
    return mask;
}

__device__ __forceinline__ unsigned int warp_reduce_sum(unsigned int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void count_gliders_kernel(
    const unsigned char* __restrict__ cells,
    int width,
    int height,
    unsigned long long* __restrict__ count) {
    __shared__ unsigned char tile[GLIDER_BLOCK_Y + 2][GLIDER_BLOCK_X + 2];
    __shared__ unsigned int warp_sums[GLIDER_BLOCK_X * GLIDER_BLOCK_Y / 32];

    const int out_w = width - 2;
    const int out_h = height - 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int linear_tid = ty * blockDim.x + tx;

    for (int load_y = ty; load_y < blockDim.y + 2; load_y += blockDim.y) {
        const int gy = blockIdx.y * blockDim.y + load_y;
        for (int load_x = tx; load_x < blockDim.x + 2; load_x += blockDim.x) {
            const int gx = blockIdx.x * blockDim.x + load_x;
            tile[load_y][load_x] =
                (gx < width && gy < height) ? cells[(size_t)gy * (size_t)width + (size_t)gx] : 0u;
        }
    }

    __syncthreads();

    unsigned int found = 0;
    if (x < out_w && y < out_h) {
        const unsigned short mask = window_mask_from_shared(tile, tx, ty);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (mask == c_glider_masks[i]) {
                found = 1;
                break;
            }
        }
    }

    const int lane = linear_tid % 32;
    const int warp_id = linear_tid / 32;
    const unsigned int warp_sum = warp_reduce_sum(found);

    if (lane == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        unsigned int val =
            (lane < (GLIDER_BLOCK_X * GLIDER_BLOCK_Y / 32)) ? warp_sums[lane] : 0u;
        val = warp_reduce_sum(val);
        if (lane == 0 && val != 0u) {
            atomicAdd(count, (unsigned long long)val);
        }
    }
}

uint64_t cuda_countGliders(const unsigned char* cells, const size_t width, const size_t height) {
    if (cells == NULL || width < 3 || height < 3) {
        return 0;
    }

    const size_t cells_bytes = width * height * sizeof(unsigned char);
    unsigned long long h_count = 0;
    unsigned short glider_masks[16];

    build_glider_masks(glider_masks);

    ensure_u8_buffer(&cached_cells, &cached_cells_bytes, cells_bytes, "cudaMalloc cached_cells");
    ensure_ull_buffer(&cached_count, &cached_count_bytes, sizeof(unsigned long long), "cudaMalloc cached_count");

    cuda_check(cudaMemcpy(cached_cells, cells, cells_bytes, cudaMemcpyHostToDevice), "cudaMemcpy cells");
    cuda_check(cudaMemcpyToSymbol(c_glider_masks, glider_masks, sizeof(glider_masks)), "cudaMemcpyToSymbol glider masks");
    cuda_check(cudaMemset(cached_count, 0, sizeof(unsigned long long)), "cudaMemset count");

    const dim3 block(GLIDER_BLOCK_X, GLIDER_BLOCK_Y);
    const dim3 grid(
        (unsigned int)(((width - 2) + block.x - 1u) / block.x),
        (unsigned int)(((height - 2) + block.y - 1u) / block.y));

    count_gliders_kernel << <grid, block >> > (cached_cells, (int)width, (int)height, cached_count);
    cuda_check(cudaGetLastError(), "count_gliders_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "count_gliders_kernel sync");

    cuda_check(cudaMemcpy(&h_count, cached_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "cudaMemcpy count");

    return (uint64_t)h_count;
}

/* Histogram */
__global__ void histogram_kernel(
    const int* __restrict__ numbers,
    int length,
    int bin_width,
    int histogram_len,
    int* __restrict__ histogram) {
    __shared__ int local_hist[HISTOGRAM_MAX_BINS];

    for (int b = threadIdx.x; b < histogram_len; b += blockDim.x) {
        local_hist[b] = 0;
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < length; i += stride) {
        const int value = __ldg(&numbers[i]);
        const int bin = value / bin_width;
        atomicAdd(&local_hist[bin], 1);
    }
    __syncthreads();

    for (int b = threadIdx.x; b < histogram_len; b += blockDim.x) {
        const int value = local_hist[b];
        if (value != 0) {
            atomicAdd(&histogram[b], value);
        }
    }
}

size_t cuda_histogram(const int* numbers, const size_t length, const int bin_width, int* output) {
    if (bin_width <= 0) {
        return 0;
    }

    const size_t histogram_len = (size_t)ceil((double)HISTOGRAM_MAX_VALUE / (double)bin_width);
    if (output == NULL) {
        return histogram_len;
    }

    memset(output, 0, histogram_len * sizeof(int));
    if (numbers == NULL || length == 0) {
        return histogram_len;
    }

    const size_t numbers_bytes = length * sizeof(int);
    const size_t histogram_bytes = histogram_len * sizeof(int);

    ensure_int_buffer(&cached_numbers, &cached_numbers_bytes, numbers_bytes, "cudaMalloc cached_numbers");
    ensure_int_buffer(&cached_histogram, &cached_histogram_bytes, histogram_bytes, "cudaMalloc cached_histogram");

    cuda_check(cudaMemcpy(cached_numbers, numbers, numbers_bytes, cudaMemcpyHostToDevice), "cudaMemcpy numbers");
    cuda_check(cudaMemset(cached_histogram, 0, histogram_bytes), "cudaMemset histogram");

    int blocks = (int)((length + HISTOGRAM_THREADS - 1u) / HISTOGRAM_THREADS);
    if (blocks < 1) blocks = 1;
    if (blocks > 1024) blocks = 1024;

    histogram_kernel << <blocks, HISTOGRAM_THREADS >> > (
        cached_numbers,
        (int)length,
        bin_width,
        (int)histogram_len,
        cached_histogram);

    cuda_check(cudaGetLastError(), "histogram_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "histogram_kernel sync");

    cuda_check(cudaMemcpy(output, cached_histogram, histogram_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy histogram");

    return histogram_len;
}

/* Emboss */
__global__ void emboss_kernel(
    const unsigned char* __restrict__ pixels,
    int width,
    int height,
    unsigned char* __restrict__ output) {
    __shared__ unsigned char tile[(EMBOSS_BLOCK_Y + 2) * (EMBOSS_BLOCK_X + 2) * 3];

    const int out_w = width - 2;
    const int out_h = height - 2;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * blockDim.x + tx;
    const int y = blockIdx.y * blockDim.y + ty;
    const int tile_w = blockDim.x + 2;

    for (int load_y = ty; load_y < blockDim.y + 2; load_y += blockDim.y) {
        const int gy = blockIdx.y * blockDim.y + load_y;
        for (int load_x = tx; load_x < blockDim.x + 2; load_x += blockDim.x) {
            const int gx = blockIdx.x * blockDim.x + load_x;
            const int tile_index = (load_y * tile_w + load_x) * 3;

            if (gx < width && gy < height) {
                const size_t global_index = ((size_t)gy * (size_t)width + (size_t)gx) * 3u;
                tile[tile_index + 0] = pixels[global_index + 0u];
                tile[tile_index + 1] = pixels[global_index + 1u];
                tile[tile_index + 2] = pixels[global_index + 2u];
            }
            else {
                tile[tile_index + 0] = 0;
                tile[tile_index + 1] = 0;
                tile[tile_index + 2] = 0;
            }
        }
    }

    __syncthreads();

    if (x >= out_w || y >= out_h) {
        return;
    }

    float pixel_sum = 0.0f;
#pragma unroll
    for (int kx = 0; kx < 3; ++kx) {
#pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            const int tile_index = ((ty + ky) * tile_w + (tx + kx)) * 3;
            const float r = (float)tile[tile_index + 0];
            const float g = (float)tile[tile_index + 1];
            const float b = (float)tile[tile_index + 2];
            const float grey = (0.2126f * r) + (0.7152f * g) + (0.0722f * b);
            pixel_sum += grey * c_emboss_kernel[ky][kx];
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
    const size_t pixels_bytes = width * height * 3u * sizeof(unsigned char);
    const size_t output_bytes = (size_t)out_w * (size_t)out_h * sizeof(unsigned char);

    ensure_u8_buffer(&cached_pixels, &cached_pixels_bytes, pixels_bytes, "cudaMalloc cached_pixels");
    ensure_u8_buffer(&cached_output, &cached_output_bytes, output_bytes, "cudaMalloc cached_output");

    cuda_check(cudaMemcpy(cached_pixels, pixels, pixels_bytes, cudaMemcpyHostToDevice), "cudaMemcpy pixels");
    cuda_check(cudaMemcpyToSymbol(c_emboss_kernel, EMBOSS_KERNEL, sizeof(EMBOSS_KERNEL)), "cudaMemcpyToSymbol emboss kernel");

    const dim3 block(EMBOSS_BLOCK_X, EMBOSS_BLOCK_Y);
    const dim3 grid(
        (unsigned int)((out_w + block.x - 1) / block.x),
        (unsigned int)((out_h + block.y - 1) / block.y));

    emboss_kernel << <grid, block >> > (cached_pixels, (int)width, (int)height, cached_output);
    cuda_check(cudaGetLastError(), "emboss_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "emboss_kernel sync");

    cuda_check(cudaMemcpy(output, cached_output, output_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy output");
}