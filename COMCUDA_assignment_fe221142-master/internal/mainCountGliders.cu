#include <cstdlib>
#include <cctype>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <random>
#include <algorithm>

#include "config.h"
#include "common.h"
#include "cpu.h"
#include "openmp.h"
#include "cuda.cuh"
#include "main.h"
#include "external/stb_image.h"

namespace {
/**
 * Structure containing the options provided by runtime arguments
 */
struct CGConfig {
    /**
      * Path to input file
      */
    char* input_file = nullptr;
    /**
      * Program will operate in benchmark mode
      * This repeats the algorithm multiple times and returns an average time
      */
    bool benchmark;
}; typedef struct CGConfig CGConfig;
/**
 * Parse the runtime args into config
 * @param argc argc from main()
 * @param argv argv from main()]
 * @param config Pointer to config structure for return value
 */
void parse_args(int argc, char** argv, CGConfig* config) {
    // Clear config struct
    *config = {};
    // Iterate over remaining args    
    int i = 3;
    char* t_arg = 0;
    for (; i < argc; i++) {
        // Make a lowercase copy of the argument
        const size_t arg_len = strlen(argv[i]) + 1;  // Add 1 for null terminating character
        if (t_arg)
            free(t_arg);
        t_arg = (char*)malloc(arg_len);
        int j = 0;
        for (; argv[i][j]; ++j) {
            t_arg[j] = tolower(argv[i][j]);
        }
        t_arg[j] = '\0';
        // Decide which arg it is
        // Benchmark
        if (!strcmp("--bench", t_arg) || !strcmp("--benchmark", t_arg) || !strcmp("-b", t_arg)) {
            config->benchmark = 1;
            continue;
        }
        // Input/Output file
        if (!strcmp(t_arg + arg_len - 5, ".png")) {
            if (!config->input_file) {
                // Allocate memory and copy
                config->input_file = (char*)malloc(arg_len);
                memcpy(config->input_file, argv[i], arg_len);
                continue;
            }
        } else {
            fprintf(stderr, ".png is the only supported input file format for the Count Gliders algorithm!\n");
            print_help(argv[0]);
        }
        fprintf(stderr, "Unexpected product of differences argument: %s\n", argv[i]);
        print_help(argv[0]);
    }
    if (!config->input_file) {
        fprintf(stderr, "An input image was not specified\n");
        print_help(argv[0]);
    }
    if (t_arg)
        free(t_arg);
}
}

void runCountGliders(int argc, char** argv, const Implementation implementation) {
    CGConfig config;
    parse_args(argc, argv, &config);

    // Inputs
    CImage input_image;
    memset(&input_image, 0, sizeof(CImage));

    // Load RGB input image
    printf("Using input file: %s%s%s\n", CONSOLE_YELLOW, config.input_file, CONSOLE_RESET);
    input_image.data = stbi_load(config.input_file, &input_image.width, &input_image.height, &input_image.channels, 0);
    if (!input_image.data) {
        fprintf(stderr, CONSOLE_RED "Unable to load image '%s', please try a different file.\n" CONSOLE_RESET, config.input_file);
        exit(EXIT_FAILURE);
    }
    if (input_image.width < 5 || input_image.height < 5) {
        fprintf(stderr, "Input image dimensions too small!\n");
        exit(EXIT_FAILURE);
    }
    if (input_image.channels > 1) {
        printf("%s%s%d%s%s\n", CONSOLE_YELLOW, "Image has ", input_image.channels, " channels, converting to 1.", CONSOLE_RESET);
        convertChannels_Nto1(&input_image);
    }
    printf("Input has dimensions: %s%d%sx%s%d%s\n", CONSOLE_YELLOW, input_image.width, CONSOLE_RESET, CONSOLE_YELLOW, input_image.height, CONSOLE_RESET);
    

    // Create result for validation
    const uint64_t countGliders_validation = cpu_countGliders(input_image.data, input_image.width, input_image.height);

    // Run student implementation
    float timing_log;
    uint64_t countGliders_result = 0;
    const int TOTAL_RUNS = config.benchmark ? BENCHMARK_RUNS : 1;
    {
        //Init for run  
        cudaEvent_t startT, stopT;
        CUDA_CALL(cudaEventCreate(&startT));
        CUDA_CALL(cudaEventCreate(&stopT));
        // Run 1 or many times
        timing_log = 0.0f;
        for (int runs = 0; runs < TOTAL_RUNS; ++runs) {
            if (TOTAL_RUNS > 1)
                printf("\r%d/%d", runs + 1, TOTAL_RUNS);
            // Run Product of Differences algorithm
            CUDA_CALL(cudaEventRecord(startT));
            CUDA_CALL(cudaEventSynchronize(startT));
            switch (implementation) {
            case CPU:
                countGliders_result = cpu_countGliders(input_image.data, input_image.width, input_image.height);
                break;
            case OPENMP:
                countGliders_result = openmp_countGliders(input_image.data, input_image.width, input_image.height);
                break;
            case CUDA:
                countGliders_result = cuda_countGliders(input_image.data, input_image.width, input_image.height);
                break;
            }
            CUDA_CALL(cudaEventRecord(stopT));
            CUDA_CALL(cudaEventSynchronize(stopT));
            // Sum timing info
            float milliseconds = 0;
            CUDA_CALL(cudaEventElapsedTime(&milliseconds, startT, stopT));
            timing_log += milliseconds;
        }
        if (TOTAL_RUNS > 1)
            printf("\n");
        // Convert timing info to average
        timing_log /= TOTAL_RUNS;

        // Cleanup timing
        cudaEventDestroy(startT);
        cudaEventDestroy(stopT);
    }

    // Validate and report
    {
        const bool FAIL = countGliders_validation != countGliders_result;
        printf("Count Gliders Result: %s" CONSOLE_RESET "\n", FAIL ? CONSOLE_RED "Fail" : CONSOLE_GREEN "Pass");
            printf("\tCPU: " CONSOLE_YELLOW "%zu" CONSOLE_RESET "\n", countGliders_validation);
            printf("\t%s: %s%zu" CONSOLE_RESET "\n", implementation_to_string(implementation), FAIL ? CONSOLE_RED : CONSOLE_GREEN, countGliders_result);
    }

    // Export output
    // Nothing to export, results are printed to stdout
    
    // Report timing information    
    printf("%s average execution timing from %d runs\n", implementation_to_string(implementation), TOTAL_RUNS);
    if (implementation == CUDA) {
        int device_id = 0;
        CUDA_CALL(cudaGetDevice(&device_id));
        cudaDeviceProp props;
        memset(&props, 0, sizeof(cudaDeviceProp));
        CUDA_CALL(cudaGetDeviceProperties(&props, device_id));
        printf("Using GPU: %s\n", props.name);
    }
#ifdef _DEBUG
    printf(CONSOLE_YELLOW "Code built as DEBUG, timing results are invalid!\n" CONSOLE_RESET);
#endif
    printf("Time: %.3fms\n", timing_log);

    // Cleanup
    stbi_image_free(input_image.data);
    free(config.input_file);
}
