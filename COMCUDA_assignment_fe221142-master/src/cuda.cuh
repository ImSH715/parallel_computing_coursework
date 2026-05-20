#ifndef __cuda_cuh__
#define __cuda_cuh__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "internal/common.h"

/**
 * @brief Counts the number of gliders in cells
 *
 * @param cells An array representing a frame of conway's game of life
 * @param width The width of frame `cells`
 * @param height The height of frame `cells`
 * @return The total number of gliders, in all stages and rotations, counted
 */
uint64_t cuda_countGliders(const unsigned char *cells, size_t width, size_t height);
/**
 * @brief Build a histogram in output, according to the provided numbers and bin_width
 *
 *
 * @param numbers An array of integer values in the inclusive range [0, HISTOGRAM_MAX_VALUE]
 * @param length The length of the array input
 * @param bin_width The divisor to test whether values in input are a factor of
 * @param output Array to store the output histogram, it may not be zero initialised
 * @return The length of the histogram stored in output
 */
size_t cuda_histogram(const int *numbers, size_t length, int bin_width, int *output);
/**
 * @brief Calculate the greyscale embossed version of the RGB input image
 *
 * @param pixels An array of pixels, where chars represent R,G,B,R,G,B,... of a 3-channel image
 * @param width The width of image `input`
 * @param height The height of image `input`
 * @param output A preallocated array to store the resulting 1-channel image, it is 2 pixels smaller in each dimension
 */
void cuda_emboss(const unsigned char *pixels, size_t width, size_t height, unsigned char* output);


/**
 * Error check function for safe CUDA API calling
 * Wrap all calls to CUDA API functions with CUDA_CALL() to catch errors on failure
 * e.g. CUDA_CALL(cudaFree(myPtr));
 * CUDA_CHECk() can also be used to perform error checking after kernel launches and async methods
 * e.g. CUDA_CHECK()
 */
#if defined(_DEBUG) || defined(D_DEBUG)
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); }
#else
#define CUDA_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUDA_CHECK() { gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }
#endif
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        if (line >= 0) {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        } else {
            fprintf(stderr, "CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
        }
        exit(EXIT_FAILURE);
    }
}

#endif // __cuda_cuh__
