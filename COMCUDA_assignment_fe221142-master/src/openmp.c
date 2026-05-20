#include "openmp.h"

#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Represent a 3x3 pattern as a 9-bit mask.
static inline unsigned short pattern_to_mask(const unsigned char pattern[3][3]) {
    unsigned short mask = 0;
    int y, x;
    for (y = 0; y < 3; ++y) {
        for (x = 0; x < 3; ++x) {
            if (pattern[y][x]) {
                mask |= (unsigned short)(1u << (y * 3 + x));
            }
        }
    }
    return mask;
}

// Rotate a 3x3 pattern mask 90 degrees clockwise.
static inline unsigned short rotate_mask_clockwise(unsigned short mask) {
    unsigned short rotated = 0;
    int y, x;
    for (y = 0; y < 3; ++y) {
        for (x = 0; x < 3; ++x) {
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

// Reflect a 3x3 pattern mask horizontally.
static inline unsigned short reflect_mask_horizontal(unsigned short mask) {
    unsigned short reflected = 0;
    int y, x;
    for (y = 0; y < 3; ++y) {
        for (x = 0; x < 3; ++x) {
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

// Build the 16 masks representing all rotations and reflections of the 2 glider patterns.
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
    int b, reflect, rot;
	// For each base mask, generate its reflection and all 4 rotations of both the original and reflected masks.
    for (b = 0; b < 2; ++b) {
        for (reflect = 0; reflect <= 1; ++reflect) {
            unsigned short m = reflect ? reflect_mask_horizontal(base_masks[b]) : base_masks[b];
            for (rot = 0; rot < 4; ++rot) {
                masks[idx++] = m;
                m = rotate_mask_clockwise(m);
            }
        }
    }
}
// Load a 3x3 window of cells into a 9-bit mask.
static inline unsigned short load_window_mask_loop(const unsigned char* row0,
    const unsigned char* row1,
    const unsigned char* row2,
    int x) {
    unsigned short mask = 0;
    const unsigned char* rows[3] = { row0, row1, row2 };
    int dy, dx;
    for (dy = 0; dy < 3; ++dy) {
        for (dx = 0; dx < 3; ++dx) {
            if (rows[dy][x + dx]) {
                mask |= (unsigned short)(1u << (dy * 3 + dx));
            }
        }
    }
    return mask;
}

// Count the number of gliders using OpenMP parallelisation.
uint64_t openmp_countGliders(const unsigned char* cells, const size_t width, const size_t height) {
    if (cells == NULL || width < 3 || height < 3) {
        return 0;
    }

    unsigned short glider_masks[16];
    build_glider_masks(glider_masks);

    const int width2 = (int)(width - 2);
    const int height2 = (int)(height - 2);
    uint64_t count = 0;
    int y;
// Parallelise the outer loop. Each iteration of the outer loop processes a different set of rows.
#pragma omp parallel for reduction(+:count) schedule(static)
    for (y = 0; y < height2; ++y) {
        const unsigned char* row0 = cells + ((size_t)y + 0u) * width;
        const unsigned char* row1 = cells + ((size_t)y + 1u) * width;
        const unsigned char* row2 = cells + ((size_t)y + 2u) * width;

        int x;
		// Load the 3x3 window into a mask and compare it against all glider masks for each position in the row.
        for (x = 0; x < width2; ++x) {
            const unsigned short window_mask = load_window_mask_loop(row0, row1, row2, x);
            int i;
            for (i = 0; i < 16; ++i) {
                if (window_mask == glider_masks[i]) {
                    ++count;
                    break;
                }
            }
        }
    }

    return count;
}
// Histogram using OpenMP parallelisation with local histogram.
size_t openmp_histogram(const int* numbers, const size_t length, const int bin_width, int* output) {
    if (bin_width <= 0) {
        return 0;
    }

    const size_t HISTOGRAM_LEN = (size_t)ceil((double)HISTOGRAM_MAX_VALUE / (double)bin_width);
    if (output == NULL) {
        return HISTOGRAM_LEN;
    }

    memset(output, 0, HISTOGRAM_LEN * sizeof(int));
    if (numbers == NULL || length == 0) {
        return HISTOGRAM_LEN;
    }

    const int num_threads = omp_get_max_threads();

    const size_t cache_line_ints = 64u / sizeof(int);
    const size_t STRIDE = ((HISTOGRAM_LEN + cache_line_ints - 1u) / cache_line_ints) * cache_line_ints;
    // Allocate a local histogram for each thread.
    int* all_local_hists = (int*)calloc((size_t)num_threads * STRIDE, sizeof(int));

    if (all_local_hists == NULL) {
        int i;
        // fallback initialisation
#pragma omp parallel for schedule(static)
        for (i = 0; i < (int)length; ++i) {
            const int bin = numbers[i] / bin_width;
            // Increment the global histogram using an atomic
#pragma omp atomic
            ++output[bin];
        }
        return HISTOGRAM_LEN;
    }

    int i;
    // Build local histograms in parallel.
#pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        int* local_hist = all_local_hists + (size_t)thread_id * STRIDE;
        // Each thread processes a different portion of the input array.
#pragma omp for schedule(static)
        for (i = 0; i < (int)length; ++i) {
            const int bin = numbers[i] / bin_width;
            ++local_hist[bin];
        }
    }

    int b;
    // Reduce local histograms into the global histogram in parallel.
#pragma omp parallel for schedule(static)
    for (b = 0; b < (int)HISTOGRAM_LEN; ++b) {
        int sum = 0;
        int t;
        for (t = 0; t < num_threads; ++t) {
            sum += all_local_hists[(size_t)t * STRIDE + (size_t)b];
        }
        output[b] = sum;
    }

    free(all_local_hists);
    return HISTOGRAM_LEN;
}

void openmp_emboss(const unsigned char* pixels, const size_t width, const size_t height, unsigned char* output) {
    static const float EMBOSS_KERNEL[3][3] = {
        {-2.0f, -1.0f, 0.0f},
        {-1.0f,  0.0f, 1.0f},
        { 0.0f,  1.0f, 2.0f}
    };

    if (pixels == NULL || output == NULL || width < 3 || height < 3) {
        return;
    }

    const size_t pixel_count = width * height;
    float* grey = (float*)malloc(pixel_count * sizeof(float));

    if (grey == NULL) {
        const int OUT_WIDTH = (int)(width - 2);
        const int OUT_HEIGHT = (int)(height - 2);
        int y;
#pragma omp parallel for schedule(static)
        for (y = 0; y < OUT_HEIGHT; ++y) {
            int x;
            for (x = 0; x < OUT_WIDTH; ++x) {
                float pixel_sum = 0.0f;
                int kx, ky;
                for (kx = 0; kx < 3; ++kx) {
                    for (ky = 0; ky < 3; ++ky) {
                        const size_t offset = (((size_t)y + (size_t)ky) * width + ((size_t)x + (size_t)kx)) * 3u;
                        const unsigned char R = pixels[offset + 0u];
                        const unsigned char G = pixels[offset + 1u];
                        const unsigned char B = pixels[offset + 2u];
                        const float grey_pixel = (0.2126f * R) + (0.7152f * G) + (0.0722f * B);
                        pixel_sum += grey_pixel * EMBOSS_KERNEL[ky][kx];
                    }
                }
                pixel_sum += 128.0f;
                if (pixel_sum < 0.0f) pixel_sum = 0.0f;
                if (pixel_sum > 255.0f) pixel_sum = 255.0f;
                output[(size_t)y * (size_t)OUT_WIDTH + (size_t)x] = (unsigned char)pixel_sum;
            }
        }
        return;
    }

    int i;
#pragma omp parallel for schedule(static)
    for (i = 0; i < (int)pixel_count; ++i) {
        const size_t offset = (size_t)i * 3u;
        const unsigned char R = pixels[offset + 0u];
        const unsigned char G = pixels[offset + 1u];
        const unsigned char B = pixels[offset + 2u];
        grey[i] = (0.2126f * R) + (0.7152f * G) + (0.0722f * B);
    }

    const int OUT_WIDTH = (int)(width - 2);
    const int OUT_HEIGHT = (int)(height - 2);
    int y;


#pragma omp parallel for schedule(static)
    for (y = 0; y < OUT_HEIGHT; ++y) {
        const size_t row0 = ((size_t)y + 0u) * width;
        const size_t row1 = ((size_t)y + 1u) * width;
        const size_t row2 = ((size_t)y + 2u) * width;
        int x;
        for (x = 0; x < OUT_WIDTH; ++x) {
            const size_t sx = (size_t)x;
            float pixel_sum = 0.0f;
            // nested loops
            int kx, ky;
            for (kx = 0; kx < 3; ++kx) {
                for (ky = 0; ky < 3; ++ky) {
                    pixel_sum += grey[((size_t)y + (size_t)ky) * width + sx + (size_t)kx] * EMBOSS_KERNEL[ky][kx];
                }
            }

            pixel_sum += 128.0f;
            if (pixel_sum < 0.0f) pixel_sum = 0.0f;
            if (pixel_sum > 255.0f) pixel_sum = 255.0f;
            output[(size_t)y * (size_t)OUT_WIDTH + sx] = (unsigned char)pixel_sum;
        }
    }

    free(grey);
}
