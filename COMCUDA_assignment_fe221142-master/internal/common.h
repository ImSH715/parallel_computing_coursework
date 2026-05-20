#ifndef __common_h__
#define __common_h__

#include <stdbool.h>
#include <stddef.h>

#include "config.h"
/**
 * This structure represents a multi-channel image
 */
struct CImage {
    /**
     * Array of pixel data of the image, 1 unsigned char per pixel channel
     * Pixels ordered left to right, top to bottom
     * There is no stride, this is a compact storage
     */
    unsigned char* data;
    /**
     * Image width and height
     */
    int width, height;
    /**
     * Number of colour channels, e.g. 1 (greyscale), 3 (rgb), 4 (rgba)
     */
    int channels;
};
typedef struct CImage CImage;
/**
 * This structure represents a single channel image (e.g. greyscale)
 * It contains the data required by the stb image read/write functions
 */
struct Image {
   /**
    * Array of pixel data of the image, 1 unsigned char per pixel channel
    * Pixels ordered left to right, top to bottom
    * There is no stride, this is a compact storage
    */
    unsigned char *data;
    /**
     * Image width and height
     */
    int width, height;
};
typedef struct Image Image;

/**
 * Common internal functions
 */
#ifdef __cplusplus
extern "C" {
#endif
/**
 * Returns true if abs(a - b) < epsilon
 * @note Useful for floating point comparisons
 */
inline bool equalsEpsilon(int a, int b, int epsilon) {
    return abs(a - b) < abs(epsilon);
}

void loadCSV(const char *input_file, void **buf, size_t *buf_elements, const char *format, size_t type_sz);
void saveCSV(const char *output_file, int *buf, size_t buf_elements);
void loadImage(const char *input_file, Image *out_image);
void saveImage(const char *output_file, Image out_image);
void convertChannels_4to3(CImage* inout_image);
void convertChannels_Nto1(CImage* inout_image);
#ifdef __cplusplus
}
#endif

#endif  // __common_h__
