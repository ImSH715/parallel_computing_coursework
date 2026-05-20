#ifndef __config_h__
#define __config_h__

#include <math.h>

/**
 * The maximum value the histogram should store
 * @note The minimum value is always 0
 */
#define HISTOGRAM_MAX_VALUE 255

/**
 * Number of runs to complete for benchmarking
 */
#define BENCHMARK_RUNS 100

/**
 * ANSI colour codes used for console output
 */
#define CONSOLE_RED "\x1b[91m"
#define CONSOLE_GREEN "\x1b[92m"
#define CONSOLE_YELLOW "\x1b[93m"
#define CONSOLE_BLUE "\x1b[94m"
#define CONSOLE_RESET "\x1b[39m"

#endif  // __config_h__
