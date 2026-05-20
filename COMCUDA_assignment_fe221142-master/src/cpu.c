#include "cpu.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

uint64_t cpu_countGliders(const unsigned char *cells, const size_t width, const size_t height) {
    // Note 1 represents white, 0 represents black (or lack of red)
    static const unsigned char GLIDER_1[3][3] =
    {{1, 0, 1},
     {1, 1, 0},
     {0, 0, 0}};
    static const unsigned char GLIDER_2[3][3] =
    {{0, 1, 0},
     {1, 0, 0},
     {1, 0, 1}};
    // Build an index, to make it convenient to access cells 2D
    const unsigned char** cell_index = (const unsigned char**)malloc(sizeof(unsigned char*) * height);
    cell_index[0] = cells;
    for (size_t i = 1; i < height; ++i) {
        cell_index[i] = cell_index[i-1] + width;
    }
    const size_t width2 = width - 2;
    const size_t height2 = height - 2;
    uint64_t count = 0;
    // Test each 3x3 group of cells
    for (size_t x = 0; x < width2; ++x) {
        for (size_t y = 0; y < height2; ++y) {
            // Process Glider 1
            for (unsigned char reflect = 0; reflect <= 1; ++reflect) {
                for (unsigned char rotate = 0; rotate <= 4; ++rotate) {
                    // x & y
                    for (int gx = 0; gx < 3; ++gx) {
                        for (int gy = 0; gy < 3; ++gy) {
                            // Calc reflection
                            int glider_x = reflect ? 2 - gx : gx;
                            int glider_y = gy;
                            // Apply rotation
                            for (int i = 0; i < rotate; ++i) {
                                const int temp_x = glider_x;
                                glider_x = glider_y;
                                glider_y = 2 - temp_x;
                            }
                            if (!((cell_index[y + gy][x + gx] && GLIDER_1[glider_y][glider_x]) || // Both are on
                                !(cell_index[y + gy][x + gx] || GLIDER_1[glider_y][glider_x]))) { // Both are off
                                // Break from the nested loop
                                goto fail_group1;
                            }
                        }
                    }
                    // Didn't break early, so we found a glider!
                    ++count;
                    // Continue an outer loop
                    goto pass_group;
                    fail_group1:;
                }
            }
            // Process Glider 2
            for (unsigned char reflect = 0; reflect <= 1; ++reflect) {
                for (unsigned char rotate = 0; rotate <= 4; ++rotate) {
                    // x & y
                    for (int gx = 0; gx < 3; ++gx) {
                        for (int gy = 0; gy < 3; ++gy) {
                            // Calc reflection
                            int glider_x = reflect ? 2 - gx : gx;
                            int glider_y = gy;
                            // Apply rotation
                            for (int i = 0; i < rotate; ++i) {
                                const int temp_x = glider_x;
                                glider_x = glider_y;
                                glider_y = 2 - temp_x;
                            }
                            if (!((cell_index[y + gy][x + gx] && GLIDER_2[glider_y][glider_x]) || // Both are on
                                !(cell_index[y + gy][x + gx] || GLIDER_2[glider_y][glider_x]))) { // Both are off
                                // Break from the nested loop
                                goto fail_group2;
                            }
                        }
                    }
                    // Didn't break early, so we found a glider!
                    ++count;
                    // Continue an outer loop
                    goto pass_group;
                    fail_group2:;
                }
            }
            pass_group:;
        }
    }
    free(cell_index);
    return count;
}

size_t cpu_histogram(const int *numbers, const size_t length, const int bin_width, int *output) {
    // Ensure output array is zero'd
    const size_t HISTOGRAM_LEN = (size_t) ceil(HISTOGRAM_MAX_VALUE / (double)bin_width);
    memset(output, 0, sizeof(int) * HISTOGRAM_LEN);
    // Construct histogram
    for (size_t i = 0; i < length; ++i) {
        ++output[numbers[i] / bin_width];
    }
    return HISTOGRAM_LEN;
}

void cpu_emboss(const unsigned char *pixels, const unsigned int width, const unsigned int height, unsigned char* output) {
    static const float EMBOSS_KERNEL[3][3] =
    {{-2, -1, 0},
     {-1,  0, 1},
     { 0,  1, 2}};
    // Iterate the output image's pixels
    const unsigned int OUT_WIDTH = width - 2;
    const unsigned int OUT_HEIGHT = height - 2;
    for (unsigned int x = 0; x < OUT_WIDTH; ++x) {
        for (unsigned int y = 0; y < OUT_HEIGHT; ++y) {
            float pixel_sum = 0;
            // Iterate the kernel to calculate the output pixel's value
            for (int kx = 0; kx < 3; ++kx) {
                for (int ky = 0; ky < 3; ++ky) {
                    // Load the source pixel's RGB components from original image
                    const unsigned int offset = (width * (y + ky) + (x + kx)) * 3;
                    const unsigned char R = pixels[offset + 0];
                    const unsigned char G = pixels[offset + 1];
                    const unsigned char B = pixels[offset + 2];
                    // Convert to greyscale
                    const float grey_pixel = (0.2126f * R) + (0.7152f * G) + (0.0722f * B);
                    // Weight and sum
                    pixel_sum += grey_pixel * EMBOSS_KERNEL[ky][kx];
                }
            }
            // Normalise, clamp and store result
            pixel_sum += 128;
            pixel_sum = pixel_sum < 0 ? 0 : pixel_sum;
            pixel_sum = pixel_sum > 255 ? 255 : pixel_sum;
            const unsigned int offset = (OUT_WIDTH * y) + x;
            output[offset] = (unsigned char)pixel_sum;
        }
    }
}
