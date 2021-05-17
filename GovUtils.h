/*
 * Author: Abdul Rafae Noor, arnoor2@illinois.edu
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>


// Multiply M1 and M2 with result in M3
inline void MultiplyFloat(float* M1, size_t r1, size_t c1,
        float* M2, size_t r2, size_t c2,
        float* M3, int perf_factor){

    int i;
    int j;
    int k;

    float upscale = perf_factor;
    for(i = 0; i < r1; i++){
        for(j = 0; j < c2; j++){
            float* res_addr = M3+(i*c2) + j; // M3[i][j]
            float res = 0.0;
            for(k = 0; k < c1;){
                float* m1_addr = M1 + (i*c1) + k;  // M1[i][k]
                float* m2_addr = M2 + (k* c2) + j; // M2[k][j]

                res += (*m1_addr)*(*m2_addr);

                k += perf_factor;

            }

            // Adjust perforated mat mul value
            *res_addr = (res* upscale);
        }
    }

}

inline void RandomizeMatrixFloat(float* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            float* res_addr = M+(i*col) + j; // M3[i][j]

            float x = (float)rand()/(float)(RAND_MAX/1.0);

            *res_addr = x;
        }
    }

}


void PrintMatrixFloat(float* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            float* res_addr = M+(i*col) + j; // M3[i][j]
            printf("%4.2f ",*res_addr);

        }
        printf("\n");
    }

}

// Multiply M1 and M2 with result in M3
inline void MultiplyInt(int* M1, size_t r1, size_t c1,
        int* M2, size_t r2, size_t c2,
        int* M3, int perf_factor){

    int i;
    int j;
    int k;

    int upscale = perf_factor;
    for(i = 0; i < r1; i++){
        for(j = 0; j < c2; j++){
            int* res_addr = M3+(i*c2) + j; // M3[i][j]
            int res = 0;
            for(k = 0; k < c1;){
                int* m1_addr = M1 + (i*c1) + k;  // M1[i][k]
                int* m2_addr = M2 + (k* c2) + j; // M2[k][j]

                res += (*m1_addr)*(*m2_addr);

                k += perf_factor;

            }

            // Adjust perforated mat mul value
            *res_addr = (res* upscale);
        }
    }

}

inline void RandomizeMatrixInt(int* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            int* res_addr = M+(i*col) + j; // M3[i][j]

            int x = rand();
            *res_addr = x;
        }
    }

}


void PrintMatrixInt(int* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            int* res_addr = M+(i*col) + j; // M3[i][j]
            printf("%d ",*res_addr);

        }
        printf("\n");
    }

}


// Multiply M1 and M2 with result in M3
inline void MultiplyChar(char* M1, size_t r1, size_t c1,
        char* M2, size_t r2, size_t c2,
        char* M3, int perf_factor){

    int i;
    int j;
    int k;

    char upscale = perf_factor;
    for(i = 0; i < r1; i++){
        for(j = 0; j < c2; j++){
            char* res_addr = M3+(i*c2) + j; // M3[i][j]
            char res = 0;
            for(k = 0; k < c1;){
                char* m1_addr = M1 + (i*c1) + k;  // M1[i][k]
                char* m2_addr = M2 + (k* c2) + j; // M2[k][j]

                res += (*m1_addr)*(*m2_addr);

                k += perf_factor;

            }

            // Adjust perforated mat mul value
            *res_addr = (res* upscale);
        }
    }

}

inline void RandomizeMatrixChar(char* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            char* res_addr = M+(i*col) + j; // M3[i][j]

            char x = rand();
            *res_addr = x;
        }
    }

}


void PrintMatrixChar(char* M, size_t row, size_t col){
    int i,j;

    for(i = 0; i < row; ++i){
        for(j=0; j <col; ++j){

            char* res_addr = M+(i*col) + j; // M3[i][j]
            printf("%c ",*res_addr);

        }
        printf("\n");
    }
}




inline void ReLUInt(int* M, size_t row, size_t col){
    for(size_t i = 0; i < row; ++i){
        for(size_t j=0; j <col; ++j){

            int* res_addr = M+(i*col) + j; // M3[i][j]

            // ReLU(x) = max(0,x)
            if(*res_addr < 0){
                *res_addr = 0;
            }

        }
    }

}

// Variadic Dense Model generation, 
// First parameter defines whether we want to 
// assign randomized weights to the model.
// The varadic arguments are the number of
// neurons in the specified layer (based of variadic
// arg idx)
//
// Input will be row major form.
//
// We propogate across the model as Data * W
int* GenerateModelInt(int RandomizeWeights, ...){
    size_t NumParameters = 1;
    size_t PrevParameters = 1;



    va_list args;
    va_start(args, RandomizeWeights);

    int FirstLayer = 1;
    
    int NumLayers = va_arg(args, int);


    size_t LayerDim;
    for(int i = 0; i < NumLayers; i++){
        LayerDim = va_arg(args, size_t);
        if(FirstLayer){
            FirstLayer = 0;
            PrevParameters = LayerDim;
            continue;
        }
        NumParameters +=  LayerDim * PrevParameters;
#ifdef GOVDEBUG
        printf("Allocating %zu Parameters for DNN Layer\n", (LayerDim * PrevParameters));
#endif
        PrevParameters = LayerDim;
    }

#ifdef GOVDEBUG
    printf("Total Number of Parameters Allocated: %zu\n",NumParameters);
#endif
    int* NetworkWeights = (int*) malloc(sizeof(int) * NumParameters);



    if(RandomizeWeights){
        // Perform randomization
        int* Offset = NetworkWeights;
        PrevParameters = 1;

        va_start(args, RandomizeWeights);
        va_arg(args, int); // Skip NumLayers

        for(int i = 0; i < NumLayers; i++){
            LayerDim = va_arg(args, size_t);
            RandomizeMatrixInt(Offset, PrevParameters, LayerDim);
            Offset += LayerDim* PrevParameters;
        }
    }

    va_end(args);

#ifdef GOVDEBUG
    printf("Finished Generating Model...\n");
#endif
    return NetworkWeights;
}


// Takes the first model to be the 
// pointer to the start of the model paramters.
//
// Second argument is the pointer to the start
// of the input argument. For space ammortization
// we allocate the entire buffers needed across layers
// ahead of time, so that across inference instances
// no additional memory need's to be allocated.
//
// The 3rd and the 4th Parameter are the shape of the input 
// the remaining arguments are the number of neurons per layer 
// in the model in the order they appear.
void InferModelInt(int PerforationFactor, ...){

#ifdef GOVDEBUG 
    printf("InferModelInt ... \n");
#endif
    va_list args;
    va_start(args, PerforationFactor);

    int NumLayers = va_arg(args, int);
    

    // Assuming the input is memcopied to the 
    // start of the buffer

    int* Model = va_arg(args, int*);
    int* InferenceBuffer  = va_arg(args,int*);


    int* DataMatrix = InferenceBuffer;
    size_t LayerDim;
    size_t NextDim;


    size_t DataOffset;
    size_t ModelOffset;

    // Input layer is a special case where the
    // first dimension is always 1. Essentially
    // the first layer is a NOP layer, since we
    // want to keep data as is.
    LayerDim = va_arg(args, size_t);
    //LayerDim = va_arg(args, size_t);


    // BATCH_SIZE = 1;

    for(int n = 1; n < NumLayers; n++){
        NextDim = va_arg(args, size_t);


        DataOffset = 1 * LayerDim;
        ModelOffset = LayerDim * NextDim;

#ifdef GOVDEBUG
        printf("LayerDim:\t%zu\n", LayerDim);
        printf("NextDim:\t%zu\n",NextDim);

        printf("DataOffset:\t%zu\n",DataOffset);
        printf("ModelOffset:\t%zu\n",ModelOffset);
#endif

        MultiplyInt(DataMatrix, 1, LayerDim, 
                    Model, LayerDim, NextDim,
                    DataMatrix + DataOffset, PerforationFactor
                    );

        DataMatrix += DataOffset;
        Model += ModelOffset;
   

        // Introducing non-linearity
        // to generalize model
        ReLUInt(DataMatrix, 1, NextDim);


        LayerDim = NextDim;


    }

#ifdef GOVDEBUG
    printf("Finished Int Inference..\n");
#endif
    va_end(args);
}


inline void ReLUFloat(float* M, size_t row, size_t col){
    for(size_t i = 0; i < row; ++i){
        for(size_t j=0; j <col; ++j){

            float* res_addr = M+(i*col) + j; // M3[i][j]

            // ReLU(x) = max(0,x)
            if(*res_addr < 0.0){
                *res_addr = 0.0;
            }

        }
    }

}
void InferModelFloat(int PerforationFactor, ...){

#ifdef GOVDEBUG 
    printf("InferModelFloat ... \n");
#endif
    va_list args;
    va_start(args, PerforationFactor);

    int NumLayers = va_arg(args, int);
    

    // Assuming the input is memcopied to the 
    // start of the buffer

    float* Model = va_arg(args, float*);
    float* InferenceBuffer  = va_arg(args,float*);


    float* DataMatrix = InferenceBuffer;
    size_t LayerDim;
    size_t NextDim;


    size_t DataOffset;
    size_t ModelOffset;

    // Input layer is a special case where the
    // first dimension is always 1. Essentially
    // the first layer is a NOP layer, since we
    // want to keep data as is.
    LayerDim = va_arg(args, size_t);
    //LayerDim = va_arg(args, size_t);


    // BATCH_SIZE = 1;

    for(int n = 1; n < NumLayers; n++){
        NextDim = va_arg(args, size_t);


        DataOffset = 1 * LayerDim;
        ModelOffset = LayerDim * NextDim;

#ifdef GOVDEBUG
        printf("LayerDim:\t%zu\n", LayerDim);
        printf("NextDim:\t%zu\n",NextDim);

        printf("DataOffset:\t%zu\n",DataOffset);
        printf("ModelOffset:\t%zu\n",ModelOffset);
#endif

        MultiplyFloat(DataMatrix, 1, LayerDim, 
                    Model, LayerDim, NextDim,
                    DataMatrix + DataOffset, PerforationFactor
                    );

        DataMatrix += DataOffset;
        Model += ModelOffset;
   

        // Introducing non-linearity
        // to generalize model
        ReLUFloat(DataMatrix, 1, NextDim);


        LayerDim = NextDim;


    }

#ifdef GOVDEBUG
    printf("Finished Float Inference..\n");
#endif
    va_end(args);
}


float* GenerateModelFloat(int RandomizeWeights, ...){
    size_t NumParameters = 1;
    size_t PrevParameters = 1;



    va_list args;
    va_start(args, RandomizeWeights);

    int FirstLayer = 1;
    
    int NumLayers = va_arg(args, int);


    size_t LayerDim;
    for(int i = 0; i < NumLayers; i++){
        LayerDim = va_arg(args, size_t);
        if(FirstLayer){
            FirstLayer = 0;
            PrevParameters = LayerDim;
            continue;
        }
        NumParameters +=  LayerDim * PrevParameters;
#ifdef GOVDEBUG
        printf("Allocating %zu Parameters for DNN Layer\n", (LayerDim * PrevParameters));
#endif
        PrevParameters = LayerDim;
    }

#ifdef GOVDEBUG
    printf("Total Number of Parameters Allocated: %zu\n",NumParameters);
#endif
    float* NetworkWeights = (float*) malloc(sizeof(float) * NumParameters);



    if(RandomizeWeights){
        // Perform randomization
        float* Offset = NetworkWeights;
        PrevParameters = 1;

        va_start(args, RandomizeWeights);
        va_arg(args, int); // Skip NumLayers

        for(int i = 0; i < NumLayers; i++){
            LayerDim = va_arg(args, size_t);
            RandomizeMatrixFloat(Offset, PrevParameters, LayerDim);
            Offset += LayerDim* PrevParameters;
        }
    }

    va_end(args);

#ifdef GOVDEBUG
    printf("Finished Generating Model...\n");
#endif
    return NetworkWeights;
}

inline void ReLUChar(char* M, size_t row, size_t col){
    for(size_t i = 0; i < row; ++i){
        for(size_t j=0; j <col; ++j){

            char* res_addr = M+(i*col) + j; // M3[i][j]

            // ReLU(x) = max(0,x)
            if(*res_addr < (char) 0){
                *res_addr = (char) 0;
            }

        }
    }

}
void InferModelChar(int PerforationFactor, ...){

#ifdef GOVDEBUG 
    printf("InferModelFloat ... \n");
#endif
    va_list args;
    va_start(args, PerforationFactor);

    int NumLayers = va_arg(args, int);
    

    // Assuming the input is memcopied to the 
    // start of the buffer

    char* Model = va_arg(args, char*);
    char* InferenceBuffer  = va_arg(args, char*);


    char* DataMatrix = InferenceBuffer;
    size_t LayerDim;
    size_t NextDim;


    size_t DataOffset;
    size_t ModelOffset;

    // Input layer is a special case where the
    // first dimension is always 1. Essentially
    // the first layer is a NOP layer, since we
    // want to keep data as is.
    LayerDim = va_arg(args, size_t);
    //LayerDim = va_arg(args, size_t);


    // BATCH_SIZE = 1;

    for(int n = 1; n < NumLayers; n++){
        NextDim = va_arg(args, size_t);


        DataOffset = 1 * LayerDim;
        ModelOffset = LayerDim * NextDim;

#ifdef GOVDEBUG
        printf("LayerDim:\t%zu\n", LayerDim);
        printf("NextDim:\t%zu\n",NextDim);

        printf("DataOffset:\t%zu\n",DataOffset);
        printf("ModelOffset:\t%zu\n",ModelOffset);
#endif

        MultiplyChar(DataMatrix, 1, LayerDim, 
                    Model, LayerDim, NextDim,
                    DataMatrix + DataOffset, PerforationFactor
                    );

        DataMatrix += DataOffset;
        Model += ModelOffset;
   

        // Introducing non-linearity
        // to generalize model
        ReLUChar(DataMatrix, 1, NextDim);


        LayerDim = NextDim;


    }

#ifdef GOVDEBUG
    printf("Finished Char Inference..\n");
#endif
    va_end(args);
}


char* GenerateModelChar(int RandomizeWeights, ...){
    size_t NumParameters = 1;
    size_t PrevParameters = 1;



    va_list args;
    va_start(args, RandomizeWeights);

    int FirstLayer = 1;
    
    int NumLayers = va_arg(args, int);


    size_t LayerDim;
    for(int i = 0; i < NumLayers; i++){
        LayerDim = va_arg(args, size_t);
        if(FirstLayer){
            FirstLayer = 0;
            PrevParameters = LayerDim;
            continue;
        }
        NumParameters +=  LayerDim * PrevParameters;
#ifdef GOVDEBUG
        printf("Allocating %zu Parameters for DNN Layer\n", (LayerDim * PrevParameters));
#endif
        PrevParameters = LayerDim;
    }

#ifdef GOVDEBUG
    printf("Total Number of Parameters Allocated: %zu\n",NumParameters);
#endif
    char* NetworkWeights = (char*) malloc(sizeof(float) * NumParameters);



    if(RandomizeWeights){
        // Perform randomization
        char* Offset = NetworkWeights;
        PrevParameters = 1;

        va_start(args, RandomizeWeights);
        va_arg(args, int); // Skip NumLayers

        for(int i = 0; i < NumLayers; i++){
            LayerDim = va_arg(args, size_t);
            RandomizeMatrixChar(Offset, PrevParameters, LayerDim);
            Offset += LayerDim* PrevParameters;
        }
    }

    va_end(args);

#ifdef GOVDEBUG
    printf("Finished Generating Model...\n");
#endif
    return NetworkWeights;
}
size_t getNumBufferParams(size_t NumLayers, ...){
    size_t numParams = 0;

    va_list args;
    va_start(args, NumLayers);

    size_t LayerDim = 0;
    for(int i = 0; i < NumLayers; i++){
        LayerDim = va_arg(args, size_t);
        numParams += LayerDim;

    }

    va_end(args);
    return numParams;

}
