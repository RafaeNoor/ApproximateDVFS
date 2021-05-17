/*
 * Author: Abdul Rafae Noor, arnoor2@illinois.edu
 *
 * ./Apx_Gov {Loop Perforation Factor} {float | int | char}
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "GovUtils.h"


struct timespec diff(struct timespec start, struct timespec end)
{
    struct timespec temp;
    if ((end.tv_nsec-start.tv_nsec)<0) {
        temp.tv_sec = end.tv_sec-start.tv_sec-1;
        temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;

    } else {
        temp.tv_sec = end.tv_sec-start.tv_sec;
        temp.tv_nsec = end.tv_nsec-start.tv_nsec;

    }
    return temp;

}

void testInt(int perf_factor){

    printf("Testing Int32 Inference \n");
    size_t NumBufferParams = getNumBufferParams(/* NumLayers */ 6, 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Number of Buffer Parameters:\t%zu\n",NumBufferParams);
#endif

    int DataBuffer[NumBufferParams];

#ifdef GOVDEBUG
    printf("Allocated Buffer!\n");
#endif
    int* Model = GenerateModelInt(/* RandomizeWeights */ 1, /* NumLayers */ 6 , 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Allocated Model and Buffer!\n");
#endif

    long repeat = 10000;
    long time_elapsed = 0;

    struct timespec start,end;

    for(int i =0; i < repeat; i++){
        RandomizeMatrixInt(DataBuffer, 1, 40);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);


        InferModelInt(perf_factor, /* NumLayers */ 6, Model, DataBuffer, 
            40 , 60, 80, 60, 40, 2);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        struct timespec diff_time = diff(start,end);

        time_elapsed += diff_time.tv_nsec;

    }

    printf("Average time elapsed in nano seconds:\t %ld\n", time_elapsed / repeat );

}

void testFloat(int perf_factor){
    printf("Testing Float Inference \n");

    size_t NumBufferParams = getNumBufferParams(/* NumLayers */ 6, 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Number of Buffer Parameters:\t%zu\n",NumBufferParams);
#endif

    float DataBuffer[NumBufferParams];

#ifdef GOVDEBUG
    printf("Allocated Buffer!\n");
#endif
    float* Model = GenerateModelFloat(/* RandomizeWeights */ 1, /* NumLayers */ 6 , 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Allocated Model and Buffer!\n");
#endif

    long repeat = 10000;
    long time_elapsed = 0;

    struct timespec start,end;

    for(int i =0; i < repeat; i++){
        RandomizeMatrixFloat(DataBuffer, 1, 40);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);


        InferModelFloat(perf_factor, /* NumLayers */ 6, Model, DataBuffer, 
            40 , 60, 80, 60, 40, 2);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        struct timespec diff_time = diff(start,end);

        time_elapsed += diff_time.tv_nsec;

    }

    printf("Average time elapsed in nano seconds:\t %ld\n", time_elapsed / repeat );

}


void testChar(int perf_factor){
    printf("Testing Char (Int8) Inference \n");

    size_t NumBufferParams = getNumBufferParams(/* NumLayers */ 6, 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Number of Buffer Parameters:\t%zu\n",NumBufferParams);
#endif

    char DataBuffer[NumBufferParams];

#ifdef GOVDEBUG
    printf("Allocated Buffer!\n");
#endif
    char* Model = GenerateModelChar(/* RandomizeWeights */ 1, /* NumLayers */ 6 , 40 , 60, 80, 60, 40, 2);



#ifdef GOVDEBUG
    printf("Allocated Model and Buffer!\n");
#endif

    long repeat = 10000;
    long time_elapsed = 0;

    struct timespec start,end;

    for(int i =0; i < repeat; i++){
        RandomizeMatrixChar(DataBuffer, 1, 40);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start);


        InferModelChar(perf_factor, /* NumLayers */ 6, Model, DataBuffer, 
            40 , 60, 80, 60, 40, 2);

        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end);
        struct timespec diff_time = diff(start,end);

        time_elapsed += diff_time.tv_nsec;

    }

    printf("Average time elapsed in nano seconds:\t %ld\n", time_elapsed / repeat );

}

int main(int argc, char** argv){


    

    enum InferenceMode {FLOAT, INT, CHAR} ;

    int perf_factor = 1;
    enum InferenceMode Mode = FLOAT;

    if(argc > 1){
        perf_factor = atoi(argv[1]);

        if(strcmp(argv[2], "float") == 0){
            Mode = FLOAT;
        } else if (strcmp(argv[2], "int") == 0){
            Mode = INT;
        } else if (strcmp(argv[2], "char") == 0){
            Mode = CHAR;
        } else {
            printf("Unknown mode specified ...\n");
            return -1;
        }
    }


    if(perf_factor <= 0){
        printf("Loop Perforation Factor must be Non-Negative!\n");
        return -1;
    }

    printf("Perforation Factor: %d\n",perf_factor);
    printf("Inference Mode: %d\n",Mode);

    switch(Mode){
        case FLOAT:
            testFloat(perf_factor);
            break;
        case INT:
            testInt(perf_factor);
            break;
        case CHAR:
            testChar(perf_factor);
            break;
    }



    return 0;
}
